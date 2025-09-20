// BEV OSINT Framework - Security-First Intelligence Platform
// Phase 0: MANDATORY SOCKS5 PROXY ENFORCEMENT

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Arc;
use tokio::sync::RwLock;
use tauri::{Manager, State};
use serde::{Deserialize, Serialize};
use anyhow::Result;

mod proxy_enforcer;
mod security;
mod osint_api;
mod osint_handlers;

use proxy_enforcer::ProxyEnforcer;
use security::SecurityManager;
use osint_handlers::OsintState;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProxyStatus {
    connected: bool,
    exit_ip: Option<String>,
    circuit_id: Option<String>,
    tor_version: Option<String>,
}

struct AppState {
    proxy_enforcer: Arc<RwLock<ProxyEnforcer>>,
    security: Arc<SecurityManager>,
    osint: Arc<OsintState>,
}

#[tauri::command]
async fn verify_proxy_status(state: State<'_, AppState>) -> Result<ProxyStatus, String> {
    let enforcer = state.proxy_enforcer.read().await;
    enforcer.get_status().await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn enforce_proxy_check(state: State<'_, AppState>) -> Result<bool, String> {
    let mut enforcer = state.proxy_enforcer.write().await;
    enforcer.validate_and_enforce().await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn rotate_circuit(state: State<'_, AppState>) -> Result<String, String> {
    let mut enforcer = state.proxy_enforcer.write().await;
    enforcer.rotate_tor_circuit().await
        .map_err(|e| e.to_string())
}

fn main() {
    // Initialize tracing for security audit logs
    tracing_subscriber::fmt::init();
    
    tauri::Builder::default()
        .setup(|app| {
            // CRITICAL: Initialize proxy enforcement BEFORE any network activity
            let proxy_enforcer = Arc::new(RwLock::new(
                ProxyEnforcer::new("127.0.0.1:9150") // Tor SOCKS5 proxy
            ));
            
            let security = Arc::new(SecurityManager::new());
            let osint = Arc::new(OsintState::new());
            
            // Validate proxy connection on startup - FAIL FAST if not connected
            let enforcer_clone = proxy_enforcer.clone();
            tauri::async_runtime::spawn(async move {
                let mut enforcer = enforcer_clone.write().await;
                if !enforcer.validate_and_enforce().await.unwrap_or(false) {
                    eprintln!("CRITICAL: Tor proxy not connected! Shutting down for OPSEC.");
                    std::process::exit(1);
                }
                println!("✓ Tor proxy validated - All traffic secured through SOCKS5");
            });
            
            app.manage(AppState {
                proxy_enforcer,
                security,
                osint,
            });
            
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            verify_proxy_status,
            enforce_proxy_check,
            rotate_circuit,
            osint_handlers::get_darknet_data,
            osint_handlers::get_crypto_data,
            osint_handlers::get_threat_intel_data,
            osint_handlers::search_address,
            osint_handlers::analyze_mixer,
            // OCR Commands
            osint_handlers::process_ocr_file,
            osint_handlers::get_ocr_status,
            osint_handlers::get_ocr_stats,
            osint_handlers::get_ocr_results,
            // Knowledge/RAG Commands
            osint_handlers::search_knowledge,
            osint_handlers::ask_document,
            osint_handlers::get_knowledge_stats,
            osint_handlers::get_knowledge_graph,
            osint_handlers::upload_to_knowledge_base
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}