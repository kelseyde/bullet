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
use viriformat::dataformat::Filter;

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
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.repeat(NUM_INPUT_BUCKETS);
                    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(255),
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            SavedFormat::id("l1w").round().quantise::<i16>(64).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(255 * 64),
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

    const STAGE1_SUPERBATCHES: usize = 100;
    const STAGE2_SUPERBATCHES: usize = 800;
    const STAGE3_SUPERBATCHES: usize = 100;
    const LR_WARMUP_BATCHES: usize = 200;
    const FIRST_WDL_FRACTION: f32 = 0.2;
    const SECOND_WDL_FRACTION: (f32, f32) = (0.4, 0.6);
    const THIRD_WDL_FRACTION: f32 = 0.8;
    const FIRST_INITIAL_LR: f32 = 0.001;
    const FIRST_FINAL_LR: f32 = 0.000027;
    const SECOND_INITIAL_LR: f32 = 0.001;
    const SECOND_FINAL_LR: f32 = 0.000027;
    const THIRD_INITIAL_LR: f32 = 0.000025;
    const THIRD_FINAL_LR: f32 = 0.0000025;

    let wdl_scheduler = wdl::Sequence {
        first: wdl::Sequence {
            first: wdl::ConstantWDL { value: FIRST_WDL_FRACTION },
            second: wdl::LinearWDL { start: SECOND_WDL_FRACTION.0, end: SECOND_WDL_FRACTION.1 },
            first_scheduler_final_superbatch: STAGE1_SUPERBATCHES,
        },
        second: wdl::ConstantWDL { value: THIRD_WDL_FRACTION, },
        first_scheduler_final_superbatch: STAGE1_SUPERBATCHES + STAGE2_SUPERBATCHES,
    };

    let lr_scheduler = lr::Sequence {
        first: lr::Sequence {
            first: lr::Warmup {
                inner: lr::LinearDecayLR {
                    initial_lr: FIRST_INITIAL_LR,
                    final_lr: FIRST_FINAL_LR,
                    final_superbatch: STAGE1_SUPERBATCHES,
                },
                warmup_batches: LR_WARMUP_BATCHES,
            },
            second: lr::LinearDecayLR {
                initial_lr: SECOND_INITIAL_LR,
                final_lr: SECOND_FINAL_LR,
                final_superbatch: STAGE2_SUPERBATCHES,
            },
            first_scheduler_final_superbatch: STAGE1_SUPERBATCHES,
        },
        second: lr::LinearDecayLR {
            initial_lr: THIRD_INITIAL_LR,
            final_lr: THIRD_FINAL_LR,
            final_superbatch: STAGE3_SUPERBATCHES,
        },
        first_scheduler_final_superbatch: STAGE1_SUPERBATCHES + STAGE2_SUPERBATCHES,
    };

    let schedule = TrainingSchedule {
        net_id: "hobbes-37-s1".to_string(),
        eval_scale: 400.0,
        steps: training_steps(1, 800),
        wdl_scheduler,
        lr_scheduler,
        save_rate: 10,
    };

    let settings = LocalSettings { threads: 12, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };
    let dataset_path = "/workspace/data/hobbes-all.vf";
    let data_loader = ViriBinpackLoader::new(dataset_path, 32768, 24, filter());

    trainer.run(&schedule, &settings, &data_loader);
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

fn filter() -> Filter {
    Filter {
        min_ply: 16,
        min_pieces: 4,
        max_eval: 31339,
        filter_tactical: true,
        filter_check: true,
        filter_castling: false,
        max_eval_incorrectness: u32::MAX,
        random_fen_skipping: true,
        random_fen_skip_probability: 0.5,
        ..Default::default()
    }
}