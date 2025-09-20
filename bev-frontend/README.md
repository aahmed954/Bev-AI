# BEV OSINT Framework Frontend
## Security-First Intelligence Platform

### ⚠️ CRITICAL SECURITY REQUIREMENTS

**ALL OPERATIONS REQUIRE TOR PROXY CONNECTION**
- The application will NOT function without Tor running on port 9150
- All network traffic is enforced through SOCKS5 proxy
- No data leaks are permitted - the app will shut down if proxy fails

### Prerequisites

1. **Tor Browser Bundle** or standalone Tor service
   - Must be running on `127.0.0.1:9150` (default Tor SOCKS5 port)
   - Verify connection: `curl --socks5 127.0.0.1:9150 https://check.torproject.org`

2. **Node.js 18+** and **Rust 1.70+**
3. **Neo4j Database** (optional, for graph analytics)

### Phase 0 Validation ✓

- [x] Tauri + SvelteKit application structure
- [x] MANDATORY SOCKS5 proxy enforcement in Rust backend
- [x] Security-first CSP headers
- [x] DOMPurify content sanitization
- [x] "Deny by default" security model

### Installation

```bash
# Install dependencies
cd bev-frontend
npm install

# Build Rust backend
cd src-tauri
cargo build --release

# Run development mode
npm run tauri:dev

# Build for production
npm run tauri:build
```

### Security Validation

The application performs these checks on startup:
1. Validates Tor proxy connection
2. Verifies exit IP through Tor network
3. Enforces CSP headers
4. Sanitizes all external content

**IF TOR IS NOT CONNECTED, THE APPLICATION WILL REFUSE TO START**

### Architecture

```
├── src-tauri/          # Rust backend (security enforcement)
│   ├── src/
│   │   ├── main.rs             # Proxy validation & enforcement
│   │   ├── proxy_enforcer.rs   # SOCKS5/Tor management
│   │   ├── security.rs         # Content sanitization
│   │   └── osint_api.rs        # Intelligence APIs
│   └── Cargo.toml
├── src/                # SvelteKit frontend
│   ├── routes/         # Application pages
│   ├── lib/
│   │   └── components/ # Intelligence dashboards
│   └── app.css         # Dark intelligence theme
└── package.json
```

### Features Implemented

- **Phase 0: Security Foundation** ✅
  - Mandatory SOCKS5 proxy enforcement
  - Cross-platform validation (Windows, macOS, Linux)
  - Security-first architecture

- **OSINT Dashboards**
  - Darknet market monitoring
  - Cryptocurrency tracking
  - Threat intelligence feeds
  - Multi-agent coordination

- **MCP Integration**
  - Model Context Protocol SDK
  - Security consent flows
  - Agent orchestration UI

### Performance Targets

- [x] <100ms UI response times
- [x] WebGL-optimized Cytoscape.js graphs
- [x] Support for 10,000+ graph nodes
- [x] Real-time WebSocket updates

### OPSEC Guidelines

1. **NEVER** run without Tor proxy
2. **ALWAYS** verify exit IP before operations
3. **ROTATE** Tor circuits regularly
4. **MONITOR** the OPSEC status bar

### License

Private - Security Research Only

---
*Built for intelligence professionals who prioritize OPSEC above all else.*