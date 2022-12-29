//hard-code audio hyperparameters

use std::error::Error;

static SAMPLE_RATE: u32 = 16000;
static N_FFT: u32 = 400;
static N_MELS: u32 = 80;
static HOP_LENGTH: u32 = 160;
static CHUNK_LENGTH: u32 = 30;
static N_SAMPLES: u32 = CHUNK_LENGTH * SAMPLE_RATE;
//480000: number of samples in a chunk
static N_FRAMES: u32 = N_SAMPLES / HOP_LENGTH;

fn load_audio(file: &str, sr: u32) -> Result<(), Box<dyn Error>> {
    todo!()
}
