use std::sync::atomic::{AtomicU64, Ordering};
use rand::{Rng, rng};
use viriformat::chess::board::Board;
use viriformat::chess::chessmove::Move;
use bullet_lib::value::loader::ViriBinpackLoader;
use bullet_lib::{
    game::inputs::{get_num_buckets, ChessBucketsMirrored},
    nn::{
        optimiser::{AdamW, AdamWParams}, InitSettings,
        Shape,
    },
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::ValueTrainerBuilder,
};
use viriformat::dataformat::{Filter, WDL};
use bullet_lib::value::loader::viribinpack::ViriFilter;

fn main() {
    // hyperparams to fiddle with
    const HL_SIZE: usize = 1280;
    const NUM_OUTPUT_BUCKETS: usize = 1;
    #[rustfmt::skip]
    const BUCKET_LAYOUT: [usize; 32] = [
         0,  1,  2,  3,
         4,  5,  6,  7,
         8,  9, 10, 11,
         8,  9, 10, 11,
        12, 12, 13, 13,
        12, 12, 13, 13,
        14, 14, 15, 15,
        14, 14, 15, 15
    ];

    const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .save_format(&[
            // merge in the factoriser weights
            SavedFormat::id("l0w")
                .add_transform(|graph, _, mut weights| {
                    let factoriser = graph.get_weights("l0f").get_dense_vals().unwrap();
                    let expanded = factoriser.repeat(NUM_INPUT_BUCKETS);
                    for (i, &j) in weights.iter_mut().zip(expanded.iter()) {
                        *i += j;
                    }
                    weights
                })
                .quantise::<i16>(255),
            SavedFormat::id("l0b").quantise::<i16>(255),
            SavedFormat::id("l1w").quantise::<i16>(64).transpose(),
            SavedFormat::id("l1b").quantise::<i16>(255 * 64),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(HL_SIZE, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, HL_SIZE);
            l0.weights = l0.weights + expanded_factoriser;

            // output layer weights
            let l1 = builder.new_affine("l1", 2 * HL_SIZE, NUM_OUTPUT_BUCKETS);

            // inference
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer)
        });

    // need to account for factoriser weight magnitudes
    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);

    let stage_1_schedule = TrainingSchedule {
        net_id: "hobbes-36-s1".to_string(),
        eval_scale: 400.0,
        steps: training_steps(1, 800),
        wdl_scheduler: wdl::Warmup { warmup_batches: 100, inner: wdl::LinearWDL { start: 0.2, end: 0.4 } },
        lr_scheduler: lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.0000081, final_superbatch: 800 },
        save_rate: 10,
    };

    let stage_2_schedule = TrainingSchedule {
        net_id: "hobbes-36-s2".to_string(),
        eval_scale: 400.0,
        steps: training_steps(1, 200),
        wdl_scheduler: wdl::ConstantWDL { value: 0.6 },
        lr_scheduler: lr::ConstantLR { value: 0.00000081 },
        save_rate: 10,
    };

    let settings = LocalSettings { threads: 12, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    let stage1_dataset_path = "/workspace/data/hobbes-all.vf";
    let stage2_dataset_path = "/workspace/data/hobbes-best.vf";

    let stage1_data_loader = ViriBinpackLoader::new(stage1_dataset_path, 32768, 24, ViriFilter::Custom(filter));
    let stage2_data_loader = ViriBinpackLoader::new(stage2_dataset_path, 32768, 24, ViriFilter::Custom(filter));

    trainer.run(&stage_1_schedule, &settings, &stage1_data_loader);
    trainer.run(&stage_2_schedule, &settings, &stage2_data_loader);
    // space needed on cluster: 1.2TB BF
}

fn training_steps(start_superbatch: usize, end_superbatch: usize) -> TrainingSteps {
    TrainingSteps {
        batch_size: 16_384,
        batches_per_superbatch: 6104,
        start_superbatch,
        end_superbatch,
    }
}

fn filter(board: &Board, mv: Move, eval: i16, wdl: f32) -> bool {
    let default_viri_filter = Filter {
        min_ply: 16,
        min_pieces: 4,
        max_eval: 31339,
        filter_tactical: true,
        filter_check: true,
        filter_castling: false,
        max_eval_incorrectness: u32::MAX,
        random_fen_skipping: true,
        random_fen_skip_probability: 0.75,
        ..Default::default()
    };
    let mut rng = rng();
    let wdl = match wdl {
        1.0 => WDL::Win,
        0.5 => WDL::Draw,
        0.0 => WDL::Loss,
        _ => unreachable!(),
    };

    !default_viri_filter.should_filter(mv, eval as i32, board, wdl, &mut rng)
        && rng.random_bool(piece_count_acceptance(board))
}

fn piece_count_acceptance(board: &Board) -> f64 {
    #[rustfmt::skip]
    const DESIRED_DISTRIBUTION: [f64; 33] = [
        0.018411966423, 0.020641545085, 0.022727271053,
        0.024669162740, 0.026467201733, 0.028121406444,
        0.029631758462, 0.030998276198, 0.032220941240,
        0.033299772000, 0.034234750067, 0.035025893853,
        0.035673184944, 0.036176641754, 0.036536245870,
        0.036752015705, 0.036823932846, 0.036752015705,
        0.036536245870, 0.036176641754, 0.035673184944,
        0.035025893853, 0.034234750067, 0.033299772000,
        0.032220941240, 0.030998276198, 0.029631758462,
        0.028121406444, 0.026467201733, 0.024669162740,
        0.022727271053, 0.020641545085, 0.018411966423,
    ];

    static PIECE_COUNT_STATS: [AtomicU64; 33] = [
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0),
    ];
    static PIECE_COUNT_TOTAL: AtomicU64 = AtomicU64::new(0);

    let pc = board.pieces.occupied().count() as usize;
    let count = PIECE_COUNT_STATS[pc].fetch_add(1, Ordering::Relaxed) + 1;
    let total = PIECE_COUNT_TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
    let frequency = count as f64 / total as f64;

    // Calculate the acceptance probability for this piece count
    let acceptance = 0.5 * DESIRED_DISTRIBUTION[pc] / frequency;
    acceptance.clamp(0., 1.)
}