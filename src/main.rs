mod activation;
mod inputs;
mod network;
mod outputs;
mod trainer;

use trainer::{lr, wdl, DirectSequentialDataLoader, LocalSettings, Trainer, TrainingSchedule, TrainingSteps};

// Network architecture settings
pub type InputFeatures = inputs::ChessBucketsMirrored;
pub type OutputBuckets = outputs::Single;
pub type Activation = activation::SCReLU;
pub const HL_SIZE: usize = 1024;

// Quantisations
pub const QA: i16 = 255;
pub const QB: i16 = 64;

/// Applicable only with `InputFeatures` option `ChessBucketsMirrored`.
/// Indexed from white POV, so index 0 corresponds to A1, 3 corresponds to D1.
#[rustfmt::skip]
pub const BUCKETS_MIRRORED: [usize; 32] = [
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
];

fn main() {
    let mut trainer = Trainer::default();
    // let mut trainer = Trainer::from_checkpoint("checkpoints/legacy-10").unwrap();

    let loader = DirectSequentialDataLoader::new(&["/Users/kelseyde/git/dan/calvin/data/calvindata.bin"]);

    let schedule = TrainingSchedule {
        net_id: "calvin1024_2b".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 800,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.4 },
        lr_scheduler: lr::LinearDecayLR { initial_lr: 0.001, final_lr: 0.000027, final_superbatch: 800 },
        save_rate: 2,
    };

    let settings = LocalSettings { threads: 10, output_directory: "checkpoints", batch_queue_size: 64 };

    trainer.run(loader, &schedule, &settings);
}
