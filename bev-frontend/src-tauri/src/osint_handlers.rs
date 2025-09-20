use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use tauri::State;
use tokio::sync::Mutex;

// Data structures for OSINT operations
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DarknetData {
    pub markets: Vec<Market>,
    pub vendors: Vec<Vendor>,
    pub products: Vec<Product>,
    pub trends: Vec<TrendData>,
    pub alerts: Vec<Alert>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Market {
    pub id: String,
    pub name: String,
    pub vendors: u32,
    pub products: u32,
    pub volume: f64,
    pub status: String,
    pub last_seen: String,
    pub escrow_balance: f64,
    pub categories: Vec<String>,
    pub trust_score: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Vendor {
    pub id: String,
    pub name: String,
    pub markets: Vec<String>,
    pub rating: f64,
    pub sales: u32,
    pub pgp_verified: bool,
    pub specialties: Vec<String>,
    pub risk_level: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Product {
    pub id: String,
    pub title: String,
    pub vendor: String,
    pub market: String,
    pub price: f64,
    pub currency: String,
    pub category: String,
    pub listing_date: String,
    pub escrow: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrendData {
    pub timestamp: String,
    pub metric: String,
    pub value: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Alert {
    pub timestamp: String,
    pub severity: String,
    pub message: String,
    pub source: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CryptoData {
    pub wallets: Vec<Wallet>,
    pub transactions: Vec<Transaction>,
    pub clusters: Vec<AddressCluster>,
    pub prices: HashMap<String, PriceData>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Wallet {
    pub address: String,
    pub label: Option<String>,
    pub balance: f64,
    pub currency: String,
    pub transactions: u32,
    pub first_seen: String,
    pub last_active: String,
    pub risk_score: f64,
    pub tags: Vec<String>,
    pub cluster: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Transaction {
    pub tx_id: String,
    pub from: String,
    pub to: String,
    pub amount: f64,
    pub currency: String,
    pub timestamp: String,
    pub fee: f64,
    pub confirmations: u32,
    pub mixer: bool,
    pub suspicious: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AddressCluster {
    pub id: String,
    pub name: String,
    pub addresses: Vec<String>,
    pub total_balance: f64,
    pub entity: Option<String>,
    pub risk_level: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PriceData {
    pub currency: String,
    pub price: f64,
    pub change_24h: f64,
    pub volume_24h: f64,
    pub market_cap: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ThreatIntelData {
    pub actors: Vec<ThreatActor>,
    pub iocs: Vec<IOC>,
    pub campaigns: Vec<Campaign>,
    pub mitre: Vec<MitreAttack>,
    pub feeds: Vec<ThreatFeed>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ThreatActor {
    pub id: String,
    pub name: String,
    pub aliases: Vec<String>,
    pub origin: String,
    pub active: bool,
    pub sophistication: String,
    pub targets: Vec<String>,
    pub ttps: Vec<String>,
    pub campaigns: Vec<String>,
    pub last_seen: String,
    pub description: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IOC {
    pub id: String,
    #[serde(rename = "type")]
    pub ioc_type: String,
    pub value: String,
    pub threat_level: f64,
    pub first_seen: String,
    pub last_seen: String,
    pub campaigns: Vec<String>,
    pub tags: Vec<String>,
    pub confidence: f64,
    pub sources: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Campaign {
    pub id: String,
    pub name: String,
    pub actor: String,
    pub status: String,
    pub start_date: String,
    pub end_date: Option<String>,
    pub targets: Vec<String>,
    pub industries: Vec<String>,
    pub techniques: Vec<String>,
    pub iocs: Vec<String>,
    pub severity: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MitreAttack {
    pub tactic: String,
    pub techniques: Vec<MitreTechnique>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MitreTechnique {
    pub id: String,
    pub name: String,
    pub used: bool,
    pub count: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ThreatFeed {
    pub id: String,
    pub name: String,
    pub enabled: bool,
    pub last_update: String,
    pub ioc_count: u32,
    pub reliability: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MixerAnalysis {
    pub address: String,
    pub mixer_score: f64,
    pub obfuscation_layers: u32,
    pub related_addresses: Vec<String>,
    pub confidence: f64,
}

// State management
pub struct OsintState {
    pub darknet_cache: Mutex<Option<DarknetData>>,
    pub crypto_cache: Mutex<Option<CryptoData>>,
    pub threat_cache: Mutex<Option<ThreatIntelData>>,
}

impl OsintState {
    pub fn new() -> Self {
        Self {
            darknet_cache: Mutex::new(None),
            crypto_cache: Mutex::new(None),
            threat_cache: Mutex::new(None),
        }
    }
}

// Tauri command handlers

#[tauri::command]
pub async fn get_darknet_data(
    filter: String,
    search: String,
    state: State<'_, OsintState>,
) -> Result<DarknetData, String> {
    // Check cache first
    let mut cache = state.darknet_cache.lock().await;
    
    if cache.is_none() {
        // Generate mock data for development
        // In production, this would fetch from actual APIs
        *cache = Some(generate_mock_darknet_data());
    }
    
    let mut data = cache.clone().unwrap();
    
    // Apply filters
    if filter != "all" {
        data.products.retain(|p| p.category == filter);
    }
    
    if !search.is_empty() {
        let search_lower = search.to_lowercase();
        data.markets.retain(|m| m.name.to_lowercase().contains(&search_lower));
        data.vendors.retain(|v| v.name.to_lowercase().contains(&search_lower));
        data.products.retain(|p| p.title.to_lowercase().contains(&search_lower));
    }
    
    Ok(data)
}

#[tauri::command]
pub async fn get_crypto_data(
    currency: String,
    time_range: String,
    state: State<'_, OsintState>,
) -> Result<CryptoData, String> {
    // Check cache first
    let mut cache = state.crypto_cache.lock().await;
    
    if cache.is_none() {
        *cache = Some(generate_mock_crypto_data());
    }
    
    let mut data = cache.clone().unwrap();
    
    // Filter by currency
    if currency != "ALL" {
        data.wallets.retain(|w| w.currency == currency);
        data.transactions.retain(|t| t.currency == currency);
    }
    
    // Filter by time range (simplified for mock)
    // In production, this would properly filter based on timestamps
    
    Ok(data)
}

#[tauri::command]
pub async fn get_threat_intel_data(
    time_range: String,
    filter: Value,
    state: State<'_, OsintState>,
) -> Result<ThreatIntelData, String> {
    // Check cache first
    let mut cache = state.threat_cache.lock().await;
    
    if cache.is_none() {
        *cache = Some(generate_mock_threat_data());
    }
    
    let mut data = cache.clone().unwrap();
    
    // Apply filters from the filter object
    if let Some(ioc_type) = filter.get("iocType").and_then(|v| v.as_str()) {
        if ioc_type != "all" {
            data.iocs.retain(|i| i.ioc_type == ioc_type);
        }
    }
    
    if let Some(severity) = filter.get("severity").and_then(|v| v.as_str()) {
        if severity != "all" {
            data.campaigns.retain(|c| c.severity == severity);
        }
    }
    
    if let Some(search) = filter.get("search").and_then(|v| v.as_str()) {
        if !search.is_empty() {
            let search_lower = search.to_lowercase();
            data.actors.retain(|a| a.name.to_lowercase().contains(&search_lower));
            data.campaigns.retain(|c| c.name.to_lowercase().contains(&search_lower));
        }
    }
    
    Ok(data)
}

#[tauri::command]
pub async fn search_address(
    address: String,
    state: State<'_, OsintState>,
) -> Result<Wallet, String> {
    // In production, this would query blockchain APIs
    // For now, return a mock wallet
    Ok(Wallet {
        address: address.clone(),
        label: Some("Searched Wallet".to_string()),
        balance: 1.234,
        currency: "BTC".to_string(),
        transactions: 42,
        first_seen: "2024-01-01T00:00:00Z".to_string(),
        last_active: chrono::Utc::now().to_rfc3339(),
        risk_score: 35.0,
        tags: vec!["searched".to_string()],
        cluster: None,
    })
}

#[tauri::command]
pub async fn analyze_mixer(
    address: String,
) -> Result<MixerAnalysis, String> {
    // Mock mixer analysis
    // In production, this would run sophisticated analysis
    Ok(MixerAnalysis {
        address: address.clone(),
        mixer_score: 65.5,
        obfuscation_layers: 3,
        related_addresses: vec![
            format!("related_{}_1", &address[0..8]),
            format!("related_{}_2", &address[0..8]),
        ],
        confidence: 78.3,
    })
}

// Helper functions to generate mock data
fn generate_mock_darknet_data() -> DarknetData {
    DarknetData {
        markets: vec![
            Market {
                id: "alpha3".to_string(),
                name: "AlphaBay3".to_string(),
                vendors: 2341,
                products: 45123,
                volume: 12.5,
                status: "active".to_string(),
                last_seen: chrono::Utc::now().to_rfc3339(),
                escrow_balance: 523.7,
                categories: vec!["drugs".to_string(), "fraud".to_string()],
                trust_score: 8.5,
            },
        ],
        vendors: vec![
            Vendor {
                id: "vendor_1".to_string(),
                name: "CryptoKing".to_string(),
                markets: vec!["alpha3".to_string()],
                rating: 4.5,
                sales: 1234,
                pgp_verified: true,
                specialties: vec!["drugs".to_string()],
                risk_level: "medium".to_string(),
            },
        ],
        products: vec![
            Product {
                id: "product_1".to_string(),
                title: "Test Product".to_string(),
                vendor: "vendor_1".to_string(),
                market: "alpha3".to_string(),
                price: 0.05,
                currency: "BTC".to_string(),
                category: "drugs".to_string(),
                listing_date: chrono::Utc::now().to_rfc3339(),
                escrow: true,
            },
        ],
        trends: vec![],
        alerts: vec![],
    }
}

fn generate_mock_crypto_data() -> CryptoData {
    let mut prices = HashMap::new();
    prices.insert("BTC".to_string(), PriceData {
        currency: "BTC".to_string(),
        price: 45000.0,
        change_24h: 2.5,
        volume_24h: 1000000000.0,
        market_cap: 900000000000.0,
    });
    
    CryptoData {
        wallets: vec![
            Wallet {
                address: "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa".to_string(),
                label: Some("Genesis Block".to_string()),
                balance: 50.0,
                currency: "BTC".to_string(),
                transactions: 1,
                first_seen: "2009-01-03T00:00:00Z".to_string(),
                last_active: "2009-01-03T00:00:00Z".to_string(),
                risk_score: 0.0,
                tags: vec!["genesis".to_string()],
                cluster: None,
            },
        ],
        transactions: vec![],
        clusters: vec![],
        prices,
    }
}

fn generate_mock_threat_data() -> ThreatIntelData {
    ThreatIntelData {
        actors: vec![
            ThreatActor {
                id: "apt28".to_string(),
                name: "APT28".to_string(),
                aliases: vec!["Fancy Bear".to_string()],
                origin: "Russia".to_string(),
                active: true,
                sophistication: "advanced".to_string(),
                targets: vec!["Government".to_string()],
                ttps: vec!["T1566".to_string()],
                campaigns: vec!["campaign_1".to_string()],
                last_seen: chrono::Utc::now().to_rfc3339(),
                description: "Advanced persistent threat group".to_string(),
            },
        ],
        iocs: vec![
            IOC {
                id: "ioc_1".to_string(),
                ioc_type: "ip".to_string(),
                value: "192.168.1.1".to_string(),
                threat_level: 75.0,
                first_seen: chrono::Utc::now().to_rfc3339(),
                last_seen: chrono::Utc::now().to_rfc3339(),
                campaigns: vec!["campaign_1".to_string()],
                tags: vec!["malware".to_string()],
                confidence: 85.0,
                sources: vec!["OSINT".to_string()],
            },
        ],
        campaigns: vec![],
        mitre: vec![],
        feeds: vec![],
    }
}

// OCR Processing Commands
#[tauri::command]
pub async fn process_ocr_file(
    job_id: String,
    filename: String,
    file_data: String,
    engines: Vec<String>,
    options: serde_json::Value,
) -> Result<String, String> {
    let client = reqwest::Client::new();
    
    let payload = serde_json::json!({
        "job_id": job_id,
        "filename": filename,
        "file_data": file_data,
        "engines": engines,
        "options": options
    });
    
    match client
        .post("http://localhost:3020/ocr/process")
        .json(&payload)
        .send()
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                Ok(job_id)
            } else {
                Err(format!("OCR service error: {}", response.status()))
            }
        }
        Err(e) => Err(format!("Failed to connect to OCR service: {}", e)),
    }
}

// Knowledge/RAG Processing Commands
#[tauri::command]
pub async fn search_knowledge(
    query: String,
    similarity_threshold: f64,
    max_results: u32,
    vector_db: String,
    include_metadata: bool,
) -> Result<serde_json::Value, String> {
    let client = reqwest::Client::new();
    
    let payload = serde_json::json!({
        "query": query,
        "similarity_threshold": similarity_threshold,
        "max_results": max_results,
        "vector_db": vector_db,
        "include_metadata": include_metadata
    });
    
    match client
        .post("http://localhost:3021/knowledge/search")
        .json(&payload)
        .send()
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                response.json().await.map_err(|e| e.to_string())
            } else {
                Err(format!("Knowledge service error: {}", response.status()))
            }
        }
        Err(e) => Err(format!("Failed to connect to knowledge service: {}", e)),
    }
}

#[tauri::command]
pub async fn ask_document(
    session_id: String,
    question: String,
    document_id: Option<String>,
    options: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let client = reqwest::Client::new();
    
    let payload = serde_json::json!({
        "session_id": session_id,
        "question": question,
        "document_id": document_id,
        "options": options
    });
    
    match client
        .post("http://localhost:3021/knowledge/ask")
        .json(&payload)
        .send()
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                response.json().await.map_err(|e| e.to_string())
            } else {
                Err(format!("RAG service error: {}", response.status()))
            }
        }
        Err(e) => Err(format!("Failed to connect to RAG service: {}", e)),
    }
}

#[tauri::command]
pub async fn get_knowledge_stats() -> Result<serde_json::Value, String> {
    let client = reqwest::Client::new();
    
    match client
        .get("http://localhost:3021/knowledge/stats")
        .send()
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                response.json().await.map_err(|e| e.to_string())
            } else {
                Err(format!("Knowledge service error: {}", response.status()))
            }
        }
        Err(e) => Err(format!("Failed to connect to knowledge service: {}", e)),
    }
}

#[tauri::command]
pub async fn get_knowledge_graph(
    vector_db: String,
    include_relationships: bool,
    max_nodes: u32,
    similarity_threshold: f64,
) -> Result<serde_json::Value, String> {
    let client = reqwest::Client::new();
    
    let payload = serde_json::json!({
        "vector_db": vector_db,
        "include_relationships": include_relationships,
        "max_nodes": max_nodes,
        "similarity_threshold": similarity_threshold
    });
    
    match client
        .post("http://localhost:3021/knowledge/graph")
        .json(&payload)
        .send()
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                response.json().await.map_err(|e| e.to_string())
            } else {
                Err(format!("Knowledge graph service error: {}", response.status()))
            }
        }
        Err(e) => Err(format!("Failed to connect to knowledge graph service: {}", e)),
    }
}

#[tauri::command]
pub async fn upload_to_knowledge_base(
    filename: String,
    file_data: String,
    vector_db: String,
    extract_entities: bool,
    build_graph: bool,
) -> Result<String, String> {
    let client = reqwest::Client::new();
    
    let payload = serde_json::json!({
        "filename": filename,
        "file_data": file_data,
        "vector_db": vector_db,
        "extract_entities": extract_entities,
        "build_graph": build_graph
    });
    
    match client
        .post("http://localhost:3021/knowledge/upload")
        .json(&payload)
        .send()
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                let result: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
                Ok(result["upload_id"].as_str().unwrap_or("unknown").to_string())
            } else {
                Err(format!("Knowledge upload service error: {}", response.status()))
            }
        }
        Err(e) => Err(format!("Failed to connect to knowledge upload service: {}", e)),
    }
}

// Additional Infrastructure Commands
#[tauri::command]
pub async fn get_database_statuses() -> Result<serde_json::Value, String> {
    // Mock database statuses for development
    let statuses = serde_json::json!([
        {
            "id": "postgresql",
            "name": "PostgreSQL",
            "type": "postgresql",
            "host": "172.21.0.2",
            "port": 5432,
            "status": "connected",
            "version": "14.9",
            "size": 8.2,
            "connections": 15,
            "maxConnections": 100,
            "uptime": 72,
            "lastCheck": chrono::Utc::now().to_rfc3339(),
            "metrics": {
                "cpu": 25.4,
                "memory": 67.8,
                "disk": 34.2,
                "queries_per_second": 145.7
            }
        }
    ]);
    
    Ok(statuses)
}

#[tauri::command]
pub async fn execute_database_query(
    database_id: String,
    query: String,
    index: Option<String>,
) -> Result<serde_json::Value, String> {
    let client = reqwest::Client::new();
    
    let payload = serde_json::json!({
        "database_id": database_id,
        "query": query,
        "index": index
    });
    
    match client
        .post("http://localhost:3010/database/query")
        .json(&payload)
        .send()
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                response.json().await.map_err(|e| e.to_string())
            } else {
                Err(format!("Database query error: {}", response.status()))
            }
        }
        Err(e) => Err(format!("Failed to connect to database service: {}", e)),
    }
}

#[tauri::command]
pub async fn get_system_performance() -> Result<serde_json::Value, String> {
    let metrics = serde_json::json!({
        "overallHealth": "healthy",
        "totalServices": 70,
        "onlineServices": 68,
        "cpuUsage": 34.7,
        "memoryUsage": 42.1,
        "diskUsage": 23.8,
        "networkThroughput": 128.4,
        "activeConnections": 1247,
        "responseTime": 45.2
    });
    
    Ok(metrics)
}

#[tauri::command]
pub async fn get_analyzer_statuses() -> Result<serde_json::Value, String> {
    let statuses = serde_json::json!([
        {
            "id": "breach",
            "name": "Breach Database Analyzer",
            "status": "online",
            "activeJobs": 2,
            "totalJobs": 156,
            "successRate": 94.2,
            "avgResponseTime": 1247.3,
            "lastUsed": chrono::Utc::now().to_rfc3339(),
            "capabilities": ["dehashed", "snusbase", "hibp"]
        }
    ]);
    
    Ok(statuses)
}

#[tauri::command]
pub async fn get_ocr_status(job_id: String) -> Result<serde_json::Value, String> {
    let client = reqwest::Client::new();
    
    match client
        .get(&format!("http://localhost:3020/ocr/status/{}", job_id))
        .send()
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                response.json().await.map_err(|e| e.to_string())
            } else {
                Err(format!("OCR service error: {}", response.status()))
            }
        }
        Err(e) => Err(format!("Failed to connect to OCR service: {}", e)),
    }
}

#[tauri::command]
pub async fn get_ocr_stats() -> Result<serde_json::Value, String> {
    let client = reqwest::Client::new();
    
    match client
        .get("http://localhost:3020/ocr/stats")
        .send()
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                response.json().await.map_err(|e| e.to_string())
            } else {
                Err(format!("OCR service error: {}", response.status()))
            }
        }
        Err(e) => Err(format!("Failed to connect to OCR service: {}", e)),
    }
}

