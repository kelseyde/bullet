use bullet_core::optimiser::adam::AdamWParams;
use bullet_lib::default::loader;
use bullet_lib::nn::optimiser::AdamW;
use bullet_lib::{
    game::inputs::{get_num_buckets, ChessBucketsMirrored},
    nn::{
        InitSettings, Shape,
    },
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::ValueTrainerBuilder,
};
use bullet_lib::default::loader::DirectSequentialDataLoader;

fn main() {
    // hyperparams to fiddle with
    const HL_SIZE: usize = 1024;
    const NUM_OUTPUT_BUCKETS: usize = 1;
    #[rustfmt::skip]
    const BUCKET_LAYOUT: [usize; 32] = [
        0, 0, 1, 1,
        2, 2, 3, 3,
        4, 4, 4, 4,
        4, 4, 4, 4,
        4, 4, 4, 4,
        5, 5, 5, 5,
        5, 5, 5, 5,
        5, 5, 5, 5,
    ];

    const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        // .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
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

    let schedule = TrainingSchedule {
        net_id: "hobbes-20".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 400,
        },
        wdl_scheduler: wdl::Warmup { warmup_batches: 100, inner: wdl::LinearWDL { start: 0.2, end: 0.4 } },
        lr_scheduler: lr::CosineDecayLR {initial_lr: 0.001, final_lr: 0.0000081, final_superbatch: 400},
        save_rate: 10,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    let data_loader = DirectSequentialDataLoader::new(&["/workspace/hobbes-6-to-16-shuffled.bin"]);

    trainer.run(&schedule, &settings, &data_loader);
}
