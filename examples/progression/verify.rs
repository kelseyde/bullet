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
use bullet_lib::game::outputs::MaterialCount;

const L1: usize = 1792;
const L2: usize = 16;
const L3: usize = 32;
const SCALE: i32 = 400;
const Q0: i16 = 255;
const Q1: i16 = 128;
const Q: i16 = 64;
const INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);
const OUTPUT_BUCKETS: usize = 8;

const FT_SHIFT: usize = 8;
const FT_SHIFT_SCALE: f32 = Q0 as f32 / ((1 << FT_SHIFT) as f32);
const I8_RANGE: f32 = i8::MAX as f32 / (Q1 as f32);
const L1_RANGE: f32 = I8_RANGE * FT_SHIFT_SCALE * FT_SHIFT_SCALE;

#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  8,  9,  9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
];

fn main() {
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(|builder, mut weights| {
                    let factoriser = builder.get("l0f").values;
                    let expanded = factoriser.repeat(INPUT_BUCKETS);

                    for (i, &j) in weights.iter_mut().zip(expanded.iter()) {
                        *i += j;
                    }

                    weights
                })
                .round()
                .quantise::<i16>(Q0),
            SavedFormat::id("l0b").round().quantise::<i16>(Q0),
            SavedFormat::id("l1w")
                .transform(|_, mut weights| {
                    for i in weights.iter_mut() {
                        *i /= FT_SHIFT_SCALE * FT_SHIFT_SCALE;
                    }
                    weights
                })
                .round()
                .quantise::<i8>(Q1),
            SavedFormat::id("l1b").round().quantise::<i32>(Q as i32),
            SavedFormat::id("l2w").round().quantise::<i32>(Q as i32),
            SavedFormat::id("l2b").round().quantise::<i32>((Q as i32).pow(3)),
            SavedFormat::id("l3w").round().quantise::<i32>(Q as i32),
            SavedFormat::id("l3b").round().quantise::<i32>((Q as i32).pow(4)),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(L1, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * INPUT_BUCKETS, L1);
            l0.weights = l0.weights + expanded_factoriser;

            // output layer weights
            let l1 = builder.new_affine("l1", L1, OUTPUT_BUCKETS * L2);
            let l2 = builder.new_affine("l2", L2 * 2, OUTPUT_BUCKETS * L3);
            let l3 = builder.new_affine("l3", L3, OUTPUT_BUCKETS);

            // inference
            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let l0_out = stm_hidden.concat(ntm_hidden);

            let l1_out = l1.forward(l0_out).select(output_buckets);
            let hl2 = l1_out.concat(l1_out.abs_pow(2.0)).crelu();

            let l2_out = l2.forward(hl2).select(output_buckets);
            let hl3 = l2_out.crelu();

            l3.forward(hl3).select(output_buckets)
        });

    let l0_clip = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", l0_clip);
    trainer.optimiser.set_params_for_weight("l0f", l0_clip);

    let l1_clip = AdamWParams { max_weight: L1_RANGE, min_weight: -L1_RANGE, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l1w", l1_clip);

    trainer.load_from_checkpoint("checkpoints/hobbes-41-s2-200");

    let eval = 400.0 * trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.0");
    println!("Eval: {eval:.3}cp");

    let eval = 400.0 * trainer.eval("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 | 0 | 0.0");
    println!("Eval: {eval:.3}cp");

    let eval = 400.0 * trainer.eval("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1 | 0 | 0.0");
    println!("Eval: {eval:.3}cp");
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