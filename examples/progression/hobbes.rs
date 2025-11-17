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

    let stage_1_schedule = TrainingSchedule {
        net_id: "hobbes-37-s1".to_string(),
        eval_scale: 400.0,
        steps: training_steps(1, 800),
        wdl_scheduler: wdl::Warmup { warmup_batches: 100, inner: wdl::LinearWDL { start: 0.2, end: 0.4 } },
        lr_scheduler: lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.0000081, final_superbatch: 800 },
        save_rate: 10,
    };

    let stage_2_schedule = TrainingSchedule {
        net_id: "hobbes-37-s2".to_string(),
        eval_scale: 400.0,
        steps: training_steps(1, 400),
        wdl_scheduler: wdl::LinearWDL { start: 0.6 , end: 0.8 },
        lr_scheduler: lr::LinearDecayLR { initial_lr: 0.00000081, final_lr: 0.00000040, final_superbatch: 400 },
        save_rate: 10,
    };

    let settings = LocalSettings { threads: 12, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    let stage1_dataset_path = "/workspace/data/hobbes-all.vf";
    let stage2_dataset_path = "/workspace/data/hobbes-best.vf";

    let stage1_data_loader = ViriBinpackLoader::new(stage1_dataset_path, 32768, 24, filter());
    let stage2_data_loader = ViriBinpackLoader::new(stage2_dataset_path, 32768, 24, filter());

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