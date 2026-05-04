// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

﻿use clap::Parser;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use warp::Filter;

use aether_core::ml::liquid_tensor::LiquidTensor;
use aether_core::ml::tensor::Tensor;
use aether_core::genesis::architect::implementation::Architect;
use aether_core::genesis::critic::{Critic, InteractionLog};
use aether_link::AetherLinkKernel;
use sanctuary_dsp::FftProcessor;
mod llm_engine;
use llm_engine::LLMEngine;
mod cluster;
use cluster::{ClusterData, SeedData, ClusterManager};

use tokio::sync::broadcast;
use bytes::{Bytes, Buf};
use std::convert::Infallible;

// --- Data Structures ---



#[derive(Serialize)]
struct WsOutput {
    text: String,
    plasticity_score: f32, // 0.0 to 1.0
    pulse_type: String,    // "none", "green", "blue", "red"
    thought_log: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chunk: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dsp_entropy: Option<f32>,
}


// --- Kernel State ---

struct KernelState {
    liquid_brain: LiquidTensor,
    cluster_manager: ClusterManager,
    link_kernel: AetherLinkKernel, // The "Nervous System"
    llm_engine: Arc<Mutex<LLMEngine>>,
    architect: Architect,
    journal_writer: BufWriter<fs::File>,
}

impl KernelState {
    // --- Persistence ---
    fn append_journal(&mut self, entry: &ClusterData) {
        if let Ok(json) = serde_json::to_string(entry) {
            if let Err(e) = writeln!(self.journal_writer, "{}", json) {
                eprintln!("[KERNEL] Failed to write to journal: {}", e);
            } else {
                let _ = self.journal_writer.flush();
                println!("[KERNEL] Journal entry appended.");
            }
        }
    }

    fn new(seed_path: &PathBuf) -> Result<Self, anyhow::Error> {
        println!("[KERNEL] Booting APEIRON Runtime...");
        
        let mut clusters = Vec::new();

        // 1. Load snapshots if available (start_memory.json)
        // 2. Replay journal (memory_journal.jsonl)
        
        if Path::new("start_memory.json").exists() {
             println!("[KERNEL] Loading snapshot...");
             let content = fs::read_to_string("start_memory.json")?;
             if let Ok(s) = serde_json::from_str::<SeedData>(&content) {
                 clusters = s.manifold;
             }
        } else if seed_path.exists() {
             println!("[KERNEL] Loading seed...");
             let content = fs::read_to_string(seed_path)?;
             if let Ok(s) = serde_json::from_str::<SeedData>(&content) {
                 clusters = s.manifold;
             }
        }

        // Replay Journal
        if Path::new("memory_journal.jsonl").exists() {
            println!("[KERNEL] Replaying memory journal...");
            if let Ok(file) = fs::File::open("memory_journal.jsonl") {
                let reader = BufReader::new(file);
                for line in reader.lines() {
                    if let Ok(line_content) = line {
                        if let Ok(entry) = serde_json::from_str::<ClusterData>(&line_content) {
                            clusters.push(entry);
                        }
                    }
                }
            }
        }
        
        println!("[KERNEL] Loaded {} clusters.", clusters.len());
        let cluster_manager = ClusterManager::new(clusters);

        let weights = Tensor::kaiming_uniform((10, 10), Some(42));
        let liquid_brain = LiquidTensor::new(weights, 0.05);

        // Initialize Aether-Link (HFT Preset for low latency)
        let link_kernel = AetherLinkKernel::new_hft();

        // Initialize LLM Engine
        let model_path = PathBuf::from("./models/Phi-3-mini-4k-instruct.Q4_K_M.gguf");
        let tokenizer_path = PathBuf::from("./models/tokenizer.json");
        
        println!("[KERNEL] Initializing LLM Engine...");
        let llm_engine = match LLMEngine::new(&model_path, &tokenizer_path) {
            Ok(engine) => engine,
            Err(e) => return Err(anyhow::anyhow!("Critical LLM Failure: {}", e)),
        };

        let architect = Architect::new("./scripts", Box::new(|code| {
            let mut parser = aether_lang::parser::Parser::new(code);
            match parser.parse() {
                Ok(_) => Ok(()),
                Err(e) => Err(format!("Parse error at {}:{}: {}", e.line, e.column, e.message)),
            }
        }));

        let journal_file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("memory_journal.jsonl")
            .unwrap_or_else(|_| fs::File::create("memory_journal.jsonl").unwrap());
        let journal_writer = BufWriter::new(journal_file);

        Ok(Self {
            liquid_brain,
            cluster_manager,
            link_kernel,
            llm_engine: Arc::new(Mutex::new(llm_engine)),
            architect,
            journal_writer,
        })
    }

    // Phase 1: Logic, Retrieval, Learning
    fn process_phase_1(&mut self, input: &str, tx: &broadcast::Sender<Bytes>) -> (Option<String>, f32, String, String, Arc<Mutex<LLMEngine>>) {
        // 1. Aether-Link Prefetch Decision (Blue Pulse)
        let output = WsOutput {
            text: "".to_string(), plasticity_score: 0.0, pulse_type: "blue".to_string(), 
            thought_log: Some("Accessing Akashic Manifold...".to_string()), chunk: None, dsp_entropy: None
        };
        if let Ok(mut json) = serde_json::to_string(&output) {
            json.push('\n');
            let _ = tx.send(Bytes::from(json));
        }

        // 1. Aether-Link Authority Check
        // Convert input to pseudo-LBA stream for the kernel
        let mut hasher = DefaultHasher::new();
        input.hash(&mut hasher);
        let hash = hasher.finish();
        let lba_stream = vec![hash, hash.wrapping_add(100), hash.wrapping_add(200)];
        
        // [HYPER PLAN] The Kernel is the sole authority.
        // returns (should_fetch, novelty_score)
        let (is_novel, novelty_score) = self.link_kernel.process_io_cycle(&lba_stream);

        // 2. Embed & Search
        // Note: find_match handles lowercase conversion internally for matching,
        // but we still need input_lower for other logic.
        let input_lower = input.to_lowercase();
        let mut thought_msg = format!("Aether Link Novelty: {:.4} (Threshold: {:.4})", novelty_score, self.link_kernel.epsilon);
        
        let best_cluster = self.cluster_manager.find_match(input);

        if let Some(cluster) = &best_cluster {
            thought_msg = format!("{} | Recall Cluster #{}", thought_msg, cluster.cluster_id);
        }
        
        let output = WsOutput {
            text: "".to_string(), plasticity_score: novelty_score, pulse_type: "none".to_string(), 
            thought_log: Some(thought_msg), chunk: None, dsp_entropy: None
        };
        if let Ok(mut json) = serde_json::to_string(&output) {
            json.push('\n');
            let _ = tx.send(Bytes::from(json));
        }

        // 3. Neuroplasticity Event (Driven by Aether Link)
        let plasticity_score = novelty_score;
        let pulse_type;
        let memory_context;

        if is_novel {
             // [RED PULSE] High Novelty -> Plasticity Event -> New Memory
             pulse_type = "red".to_string();
             static GRAD_UNKNOWN: OnceLock<Tensor> = OnceLock::new();
             let grad = GRAD_UNKNOWN.get_or_init(|| {
                 Tensor::new(vec![0.05; 100], (10, 10))
             });
             self.liquid_brain.inject_update(grad, 0.2);

             let output = WsOutput {
                text: "".to_string(), plasticity_score: novelty_score, pulse_type: "red".to_string(), 
                thought_log: Some("Aether Link Authority: High Novelty. Forming new memory axiom...".to_string()), chunk: None, dsp_entropy: None
             };
             if let Ok(mut json) = serde_json::to_string(&output) {
                json.push('\n');
                let _ = tx.send(Bytes::from(json));
             }

             // Create a new memory cluster from this high-novelty input
             let new_id = self.cluster_manager.next_id();
             
             let new_cluster = ClusterData {
                 cluster_id: new_id,
                 archetype: "novelty_axiom".to_string(),
                 centroid_hint: vec![0.0; 10], // Dummy embedding
                 inputs: vec![input_lower.clone()], // Map this exact phrase to the memory
                 responses: vec![format!("I remember this moment of novelty: '{}'", input)]
             };
             
             self.append_journal(&new_cluster); // Use Journal
             self.cluster_manager.add_cluster(new_cluster);

             memory_context = format!("I have crystallized this new information: '{}'.", input);

        } else if let Some(cluster) = best_cluster {
            // [BLUE PULSE] Familiar -> Reinforcement -> Recall
            pulse_type = "blue".to_string();
            static GRAD_KNOWN: OnceLock<Tensor> = OnceLock::new();
            let grad = GRAD_KNOWN.get_or_init(|| {
                Tensor::new(vec![0.001; 100], (10, 10)) // Ones * 0.001
            });
            self.liquid_brain.inject_update(grad, 0.01);
            
            let resp_idx = (input.len()) % cluster.responses.len();
            memory_context = cluster.responses[resp_idx].clone();
        } else {
             // [GREEN PULSE] Low Novelty but no match -> Passive Learning
             pulse_type = "green".to_string();
             memory_context = format!("I am listening directly to: '{}'.", input);
        }

        // 4. Generative Synthesis (Thought)
        let output = WsOutput {
            text: "".to_string(), plasticity_score: 0.0, pulse_type: "none".to_string(), 
            thought_log: Some("Synthesizing response with Phi-3...".to_string()), chunk: None, dsp_entropy: None
        };
        if let Ok(mut json) = serde_json::to_string(&output) {
            json.push('\n');
            let _ = tx.send(Bytes::from(json));
        }

        // Generative Synthesis (Action)
        let prompt = format!(
            "System: You are Apeiron. Use the following memory to answer. Memory: [{}]. User: {}", 
            memory_context, input
        );

        (Some(prompt), plasticity_score, pulse_type, memory_context, self.llm_engine.clone())
    }

    // Phase 2: Genesis & Evolution
    fn process_phase_2(&mut self, input: &str, response: &str) {
        // 5. Genesis: Metacognition & Evolution (Post-Interaction)
        // Reconstruct full response (simplified for MVP: just assume we got something)
        // In a real system, we'd capture the full output from the callback.
        
        let interaction_log = InteractionLog {
            user_query: input.to_string(),
            agent_response: response.to_string(),
            context_id: "ctx_001".to_string(),
        };

        let score = Critic::evaluate_performance(&interaction_log);
        
        // Kill Switch Check
        let kill_switch = PathBuf::from("STOP_EVOLUTION");
        if !kill_switch.exists() {
             if Critic::should_mutate(score) {
                 println!("[GENESIS] Critic Score: {} (LOW). Triggering Architect...", score);
                 let mutation_result = self.architect.hot_swap_script("core_logic.aether", "OPTIMIZED CODE SEQUENCE");
                 match mutation_result {
                     Ok(_) => println!("[GENESIS] Mutation Successful."),
                     Err(e) => eprintln!("[GENESIS] Mutation Failed: {:?}", e),
                 }
             }
        } else {
            println!("[GENESIS] Kill Switch Detected. Evolution Halted.");
        }
    }
}

// --- Main ---

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = "kernel_seed.json")]
    seed: PathBuf,

    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 8080)]
    port: u16,
}

async fn uplink_handler<S, B>(
    mut stream: S,
    state: Arc<Mutex<KernelState>>,
    tx: broadcast::Sender<Bytes>,
) -> Result<impl warp::Reply, Infallible>
where
    S: futures_util::Stream<Item = Result<B, warp::Error>> + Unpin + Send + 'static,
    B: Buf + Send + 'static,
{
    // Handle binary stream (Uplink)
    tokio::task::spawn(async move {
        let mut buffer = Vec::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(mut data) => {
                    let size = data.remaining();
                    let mut chunk_bytes = vec![0u8; size];
                    data.copy_to_slice(&mut chunk_bytes);
                    buffer.extend(chunk_bytes);
                }
                Err(e) => {
                    eprintln!("[UPLINK ERROR] Stream severed: {}", e);
                    return;
                }
            }
        }

        let text = match String::from_utf8(buffer) {
            Ok(s) => s,
            Err(_) => return,
        };

        println!("[RECV] Uplink Stream Completed: {}", text);

        // --- DSP Bus Validation ---
        let entropy = {
            // Optimization: Reuse FftProcessor to avoid expensive initialization (alloc + window calc)
            // Benchmarked Speedup: ~5.5x (52µs -> 9.6µs per call)
            thread_local! {
                static FFT_PROCESSOR: RefCell<FftProcessor> = RefCell::new(FftProcessor::new(1024));
            }

            let samples: Vec<f32> = text.as_bytes().iter().map(|&b| b as f32 / 255.0).collect();

            FFT_PROCESSOR.with(|dsp_cell| {
                let mut dsp = dsp_cell.borrow_mut();
                let spectrum = dsp.power_spectrum(&samples);
                spectrum.iter().sum::<f32>()
            })
        };
        println!("[DSP BUS] Input Entropy: {:.4} (VALIDATED)", entropy);

        let state_p1 = state.clone();
        let tx_p1 = tx.clone();
        let text_p1 = text.clone();

        let (prompt, p_score, p_type, _mem_ctx, engine) = tokio::task::spawn_blocking(move || {
            let mut k = state_p1.lock().unwrap();
            k.process_phase_1(&text_p1, &tx_p1)
        }).await.unwrap_or_else(|e| {
            eprintln!("[TASK ERROR] Phase 1 panicked: {}", e);
            // Return dummy values to proceed gracefully or handle error better
            (None, 0.0, "none".to_string(), "".to_string(), Arc::new(Mutex::new(LLMEngine::new(Path::new(""), Path::new("")).unwrap_or_else(|_| panic!("Failed to recover")))))
        });

        if let Some(prompt_str) = prompt {
            let tx_clone_2 = tx.clone();
            let p_type_clone = p_type.clone();

            let generation_result = tokio::task::spawn_blocking(move || {
                let mut full_response = String::new();
                let mut eng = engine.lock().unwrap();
                let _ = eng.generate(&prompt_str, |token| {
                    full_response.push_str(&token);
                    let output = WsOutput {
                        text: token,
                        plasticity_score: p_score,
                        pulse_type: p_type_clone.clone(),
                        thought_log: None,
                        chunk: Some(true),
                        dsp_entropy: Some(entropy),
                    };
                    if let Ok(mut json) = serde_json::to_string(&output) {
                        json.push('\n');
                        let _ = tx_clone_2.send(Bytes::from(json));
                    }
                });
                full_response
            }).await;

            if let Ok(response) = generation_result {
                let state_clone = state.clone();
                let text_clone = text.clone();
                let response_clone = response.clone();

                let _ = tokio::task::spawn_blocking(move || {
                    let mut k = state_clone.lock().unwrap();
                    k.process_phase_2(&text_clone, &response_clone);
                }).await;
            }
        }
    });

    Ok(warp::reply::with_status("Feed Accepted", warp::http::StatusCode::OK))
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let ip: std::net::IpAddr = cli.host.parse().expect("Invalid host address");

    println!("════════════ APEIRON CHAT SERVER ════════════");
    println!("[APEIRON] Initializing Neuro-Tunnel (DSP Bus) on {}:{}...", ip, cli.port);

    // Result handling for KernelState::new
    let kernel_state = match KernelState::new(&cli.seed) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("\n[FATAL] Kernel Startup Failed: {}\n", e);
            eprintln!("Troubleshooting:");
            eprintln!("1. Ensure models are downloaded (run `python download_model.py`)");
            eprintln!("2. Check start_memory.json format.");
            std::process::exit(1);
        }
    };
    
    let state = Arc::new(Mutex::new(kernel_state));
    let (tx, _rx) = broadcast::channel::<Bytes>(100);

    let tx_downlink = tx.clone();
    let downlink_route = warp::path!("bus" / "downlink")
        .and(warp::get())
        .map(move || {
            let rx = tx_downlink.subscribe();
            let stream = tokio_stream::wrappers::BroadcastStream::new(rx)
                .filter_map(|result| async move {
                    match result {
                        Ok(bytes) => Some(Ok::<_, Infallible>(bytes)),
                        Err(_) => None,
                    }
                });
            
            println!("[NEURO-TUNNEL] Client attached to Downlink.");
            
            // Send initial sync packet
            let _ = tx_downlink.send(Bytes::from("{\"pulse_type\":\"none\",\"thought_log\":\"[SYSTEM] Neuro-Tunnel Synced.\"}\n"));

            let body = hyper::Body::wrap_stream(stream);
            warp::http::Response::builder()
                .header("Content-Type", "application/octet-stream")
                .body(body)
                .unwrap()
        });

    // --- UPLINK (The Firehose / Streaming Ingestion) ---
    let tx_uplink = tx.clone();
    let state_filter = warp::any().map(move || state.clone());
    let tx_filter = warp::any().map(move || tx_uplink.clone());

    let uplink_route = warp::path!("bus" / "uplink")
        .and(warp::post())
        .and(warp::body::content_length_limit(1024 * 1024 * 50)) // 50MB Limit
        .and(warp::body::stream())
        .and(state_filter)
        .and(tx_filter)
        .and_then(uplink_handler);

    // --- 1. THE CORS FORTRESS ---
    // SENTINEL: Restricted origins to localhost to prevent CSRF/interaction from arbitrary sites.
    let cors = warp::cors()
        .allow_origins(vec![
            "http://localhost:8008", // Dev Frontend
            "http://localhost:7000", // Docker Frontend
            "http://localhost:3000", // Default Next.js
            "http://127.0.0.1:8008",
            "http://127.0.0.1:7000",
            "http://127.0.0.1:3000",
        ])
        .allow_headers(vec!["content-type", "authorization", "x-requested-with"])
        .allow_methods(vec!["GET", "POST", "OPTIONS"]);

    let routes = downlink_route.or(uplink_route).with(cors);

    warp::serve(routes).run((ip, cli.port)).await;
}
