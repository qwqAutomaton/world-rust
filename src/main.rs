use hound;
use std::io::Read;
mod common;
pub mod dio;
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
// fn spectrum_envelope(samples: &[f64], fs: u32, f0: &dio::F0Contour) -> Vec<Vec<f64>> {
//     Vec::new() // todo
// }
// fn aperiodic_param(
//     samples: &[f64],
//     fs: u32,
//     f0: &dio::F0Contour,
//     spec: &[Vec<f64>],
// ) -> Vec<Vec<f64>> {
//     Vec::new()
// }

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
        // todo
        // let start = std::time::Instant::now();
        // let spenv: Vec<Vec<f64>> = spectrum_envelope(&samples, fs, &f0);
        // dbg!("CheapTrick done. Time: {} ms\n", start.elapsed());
        // let start = std::time::Instant::now();
        // let apar = aperiodic_param(&samples, fs, &f0, &spenv);
        // dbg!("PLATINUM done. Time: {} ms\n", start.elapsed());
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
        // if let Err(e) = write_2darray(&spenv, &spectrum_path) {
        //     eprintln!("Fatal: Connot write spectrum into {}: {}", spectrum_path, e);
        // }
        // if let Err(e) = write_2darray(&apar, &aperiodic_path) {
        //     eprintln!(
        //         "Fatal: Connot write aperiodic parameter into {}: {}",
        //         aperiodic_path, e
        //     );
        // }
    } else {
        eprintln!("Fatal: requires 1 args, found 0.");
    }
}
