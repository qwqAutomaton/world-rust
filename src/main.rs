use hound;
use std::io::Read;
pub mod cheaptrick;
mod common;
pub mod d4c;
pub mod dio;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    // test: generate spectrum with a specified f0 contour
    #[test]
    fn spectrum_with_external_f0() {
        let wav_path = "assets/example_teto_a.wav";
        let f0_path = "assets/A.f0";
        let output_spec_path = "assets/example_teto_a.wav.se";
        let fs = 44100; // a typical sample rate

        // 1. Read wav file
        let mut reader = hound::WavReader::open(wav_path).expect("Failed to open wav file");
        let samples = reader
            .samples::<i16>()
            .map(|x| x.unwrap() as f64)
            .collect::<Vec<f64>>();

        // 2. Read f0 file
        let f0_str = fs::read_to_string(f0_path).expect("Failed to read f0 file");
        let f0_vec: Vec<f64> = f0_str
            .lines()
            .map(|line| line.trim().parse::<f64>().unwrap_or(0.0))
            .collect();

        // 3. Create F0Contour
        let frame_period = 5.0; // default from DioOption
        let tpos = (0..f0_vec.len())
            .map(|i| i as f64 * frame_period / 1000.0)
            .collect();
        let f0_contour = dio::F0Contour { f0: f0_vec, tpos };

        // 4. Calculate spectrum envelope
        let spenv = spectrum_envelope(&samples, fs, &f0_contour);

        // 5. Write to file
        let write_result = write2d_binary(&spenv, output_spec_path);

        // 6. Assert success
        assert!(write_result.is_ok(), "Failed to write spectrum file");
        println!("Test spectrum envelope written to {}", output_spec_path);
    }
}
const F0_FLOOR: f64 = 71.0;
const D4C_THRESHOLD: f64 = 0.85;
fn get_fftsize(fs: u32, f0floor: f64) -> usize {
    let size = 3.0 * fs as f64 / f0floor + 1.0;
    let log2_size = size.log2();
    2_f64.powi(1 + log2_size.floor() as i32) as usize
}
fn f0_contour(samples: &[f64], fs: u32, speed: u32) -> dio::F0Contour {
    eprintln!("Analysing F0 Contour.");
    dio::dio(
        samples,
        fs,
        &dio::DioOption {
            speed,
            ..Default::default()
        },
    )
}
fn spectrum_envelope(samples: &[f64], fs: u32, f0: &dio::F0Contour) -> Vec<Vec<f64>> {
    eprintln!("Analysing Spectrum Envelope.");
    cheaptrick::cheaptrick(
        samples,
        fs,
        &f0,
        &cheaptrick::CheapTrickOption {
            fftsize: get_fftsize(fs, F0_FLOOR),
            ..Default::default()
        },
    )
}
fn aperiodic_param(samples: &[f64], fs: u32, f0: &dio::F0Contour) -> Vec<Vec<f64>> {
    // D4COption option;
    //     InitializeD4COption(&option);

    //     // Parameters setting and memory allocation.
    //     world_parameters->aperiodicity = new double *[world_parameters->f0_length];
    //     for (int i = 0; i < world_parameters->f0_length; ++i) {
    //         world_parameters->aperiodicity[i] =
    //             new double[world_parameters->fft_size / 2 + 1];
    //     }

    //     DWORD elapsed_time = timeGetTime();
    //     // option is not implemented in this version. This is for future update.
    //     // We can use "NULL" as the argument.
    //     D4C(x, x_length, world_parameters->fs, world_parameters->time_axis,
    //         world_parameters->f0, world_parameters->f0_length,
    //         world_parameters->fft_size, &option, world_parameters->aperiodicity);

    //     std::cout << "D4C: " << timeGetTime() - elapsed_time << " [msec]" << std::endl;
    eprintln!("Analysing Aperiodic Parameters.");
    d4c::D4C(samples, fs, &f0, get_fftsize(fs, F0_FLOOR), D4C_THRESHOLD)
}

fn write2d_binary(arr: &[Vec<f64>], path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    for row in arr {
        for v in row {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if let Some(filename) = args.get(1) {
        let f0_path = format!("{}.f0", filename);
        let spectrum_path = format!("{}.se", filename);
        let aperiodic_path = format!("{}.ae", filename);
        println!("Summary:");
        println!("Wave file: {}", filename);
        println!("F0 contour will be written into: {}", f0_path);
        println!("Spectrum envelope will be written into: {}", spectrum_path);
        println!(
            "Aperiodic parameters will be written into: {}",
            aperiodic_path
        );
        println!("Continue? [yn]");
        let mut input = [0u8; 1];
        std::io::stdin().read_exact(&mut input).unwrap();
        let key = input[0].to_ascii_lowercase() as char;
        if key != 'y' {
            return;
        }
        // read from wavefile
        let mut reader = hound::WavReader::open(filename).unwrap();
        let spec = reader.spec();
        if spec.channels != 1 {
            eprintln!(
                "Fatal: stereo is not supported ({} channels found, expected 1).",
                spec.channels
            );
            return;
        }
        if spec.sample_format != hound::SampleFormat::Int {
            eprintln!("Fatal: wav format not suppoerted(IEEE float found, expected PCM)");
            return;
        }
        // wave info
        println!("{:#?}", spec);
        println!("Reading samples...");
        let samples = reader
            .samples::<i16>()
            .map(|x| x.unwrap() as f64)
            .collect::<Vec<f64>>();
        println!(
            "Length: {} samples, {:.3} seconds",
            samples.len(),
            samples.len() as f64 / spec.sample_rate as f64
        );
        println!("Analysis start...");
        let fs = spec.sample_rate;
        let start = std::time::Instant::now();
        let r = match args.get(2) {
            Some(r_str) => r_str.parse().unwrap(),
            None => 1,
        };
        let f0 = f0_contour(&samples, fs, r);
        eprintln!("DIO done (with stonemask). Time: {:?}", start.elapsed());
        let start = std::time::Instant::now();
        let spenv: Vec<Vec<f64>> = spectrum_envelope(&samples, fs, &f0);
        eprintln!("CheapTrick done. Time: {:?}", start.elapsed());
        let frame_size = spenv.get(0).map(|row| row.len()).unwrap_or(0);
        eprintln!("Frame size: {}", frame_size);
        if let Err(e) = write2d_binary(&spenv, &spectrum_path) {
            eprintln!("Fatal: Cannot write spectrum into {}: {}", spectrum_path, e);
        };
        let start = std::time::Instant::now();
        let aperiodic: Vec<Vec<f64>> = aperiodic_param(&samples, fs, &f0);
        eprintln!("D4C done. Time: {:?}", start.elapsed());
        if let Err(e) = write2d_binary(&aperiodic, &aperiodic_path) {
            eprintln!(
                "Fatal: Cannot write aperiodicity into {}: {}",
                aperiodic_path, e
            );
        };
        if let Err(e) = std::fs::write(
            &f0_path,
            f0.f0
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("\n"),
        ) {
            eprintln!("Fatal: Cannot write into F0 file {}: {}", f0_path, e);
        };
    } else {
        eprintln!("Fatal: requires 1 args, found 0.");
    }
}
