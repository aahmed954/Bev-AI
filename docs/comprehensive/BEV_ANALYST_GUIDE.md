# BEV OSINT Framework - Analyst Workflow Guide

## Overview

This guide provides comprehensive workflows and best practices for OSINT analysts using the BEV Framework. It covers investigation methodologies, tool usage patterns, data correlation techniques, and operational security practices specific to intelligence analysis.

## Table of Contents

1. [Getting Started as an Analyst](#getting-started-as-an-analyst)
2. [Investigation Methodologies](#investigation-methodologies)
3. [Core Analysis Workflows](#core-analysis-workflows)
4. [Advanced Analysis Techniques](#advanced-analysis-techniques)
5. [Data Correlation & Visualization](#data-correlation--visualization)
6. [Operational Security for Analysts](#operational-security-for-analysts)
7. [Report Generation & Documentation](#report-generation--documentation)
8. [Case Studies & Examples](#case-studies--examples)

---

## Getting Started as an Analyst

### Access and Setup

#### Initial System Access
```bash
# 1. Access the BEV web interface
Browser: https://your-bev-instance/
Default: No authentication required (single-user system)

# 2. Verify Tor proxy is active
curl -x socks5://127.0.0.1:9050 http://check.torproject.org/
# Should return: "Congratulations. This browser is configured to use Tor."

# 3. Check system health
Navigate to: https://your-bev-instance/health-check
All services should show green status
```

#### Interface Overview
```
BEV Dashboard Layout:
┌─────────────────────────────────────────────────────────────────┐
│                         BEV OSINT                              │
│  [Search] [Analysis] [Visualize] [Reports] [Monitor] [Admin]   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Quick Actions:                Active Investigations:           │
│  • New Investigation           • Case #2024-001 (Active)       │
│  • Batch Analysis             • Case #2024-002 (Complete)      │
│  • Data Import               • Case #2024-003 (Pending)       │
│  • Generate Report                                             │
│                                                                 │
│  Recent Results:              System Status:                   │
│  • Email breach found         🟢 All systems operational        │
│  • Crypto wallet linked       🟢 Tor circuits: 8 active        │
│  • Social profiles matched    🟢 Analyzers: 12 running         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Analyst Workspace Setup

#### Browser Configuration
```javascript
// Recommended browser settings for OSINT work
Settings Checklist:
✓ Disable JavaScript by default (enable per-site as needed)
✓ Block third-party cookies
✓ Disable location services
✓ Use private/incognito mode
✓ Configure proxy settings (if manual Tor usage needed)

// Browser extensions (optional)
- User Agent Switcher
- Cookie Manager
- Privacy Badger
- NoScript
```

#### Investigation Template Structure
```
Investigation Folder Structure:
/investigations/
├── YYYY-MM-DD_CaseName/
│   ├── 01_initial_intel/
│   │   ├── source_data.txt
│   │   └── target_list.csv
│   ├── 02_analysis_results/
│   │   ├── breach_data/
│   │   ├── social_media/
│   │   ├── cryptocurrency/
│   │   └── darknet/
│   ├── 03_correlation/
│   │   ├── timeline.xlsx
│   │   ├── relationship_map.json
│   │   └── attribution_analysis.md
│   ├── 04_visualization/
│   │   ├── network_graphs/
│   │   └── timeline_charts/
│   └── 05_reporting/
│       ├── executive_summary.md
│       ├── technical_report.pdf
│       └── evidence_package/
```

---

## Investigation Methodologies

### Intelligence Collection Framework

#### OSINT Collection Pyramid
```
                    ┌─────────────────┐
                    │   ATTRIBUTION   │
                    │   & ANALYSIS    │
                    └─────────┬───────┘
                              │
                    ┌─────────▼───────┐
                    │  CORRELATION &  │
                    │  VERIFICATION   │
                    └─────────┬───────┘
                              │
            ┌─────────────────▼─────────────────┐
            │        DATA ENRICHMENT           │
            │   (Social, Financial, Technical) │
            └─────────────────┬─────────────────┘
                              │
      ┌───────────────────────▼───────────────────────┐
      │              INITIAL COLLECTION               │
      │    (Breach Data, Public Records, SOCMINT)    │
      └───────────────────────────────────────────────┘
```

#### Investigation Phases

**Phase 1: Reconnaissance & Initial Collection**
```yaml
Objective: "Gather initial intelligence on targets"
Duration: "2-4 hours"
Tools: "Breach analyzers, social media scanners"
Outputs: "Target profile, initial data points"

Activities:
  - Target identification and profiling
  - Initial breach database searches
  - Social media footprint mapping
  - Public record collection
  - Domain and infrastructure analysis
```

**Phase 2: Deep Analysis & Enrichment**
```yaml
Objective: "Enrich data and identify patterns"
Duration: "4-8 hours"
Tools: "All custom analyzers, graph database"
Outputs: "Enriched datasets, relationship maps"

Activities:
  - Comprehensive breach analysis
  - Cryptocurrency transaction tracing
  - Social network analysis
  - Darknet presence investigation
  - Cross-platform correlation
```

**Phase 3: Correlation & Verification**
```yaml
Objective: "Verify findings and build attribution"
Duration: "2-4 hours"
Tools: "Neo4j, Cytoscape visualization, manual verification"
Outputs: "Verified intelligence, attribution analysis"

Activities:
  - Data cross-verification
  - Timeline construction
  - Attribution analysis
  - Confidence scoring
  - Gap identification
```

**Phase 4: Reporting & Documentation**
```yaml
Objective: "Document findings and create actionable intelligence"
Duration: "2-3 hours"
Tools: "Report generators, visualization tools"
Outputs: "Comprehensive report, evidence package"

Activities:
  - Executive summary creation
  - Technical report writing
  - Evidence package assembly
  - Recommendation development
  - Case documentation
```

### Target Analysis Methodology

#### Individual Target Analysis
```python
# Target analysis workflow
def analyze_individual_target(target_identifier):
    """
    Comprehensive individual analysis workflow
    """
    results = {}
    
    # 1. Initial collection
    results['breach_data'] = search_breach_databases(target_identifier)
    results['social_profiles'] = scan_social_media(target_identifier)
    results['public_records'] = search_public_records(target_identifier)
    
    # 2. Data enrichment
    for email in extract_emails(results):
        results['email_analysis'] = analyze_email_patterns(email)
        results['domain_analysis'] = analyze_email_domain(email)
    
    # 3. Financial analysis
    crypto_addresses = extract_crypto_addresses(results)
    results['crypto_analysis'] = trace_cryptocurrency(crypto_addresses)
    
    # 4. Correlation analysis
    results['timeline'] = build_activity_timeline(results)
    results['connections'] = find_associated_entities(results)
    
    return results
```

#### Organizational Target Analysis
```python
def analyze_organization_target(organization):
    """
    Organization-focused analysis workflow
    """
    results = {}
    
    # 1. Infrastructure analysis
    results['domains'] = enumerate_organization_domains(organization)
    results['ip_ranges'] = identify_ip_ranges(organization)
    results['certificates'] = analyze_ssl_certificates(organization)
    
    # 2. Personnel analysis
    results['employees'] = identify_employees(organization)
    results['executives'] = analyze_leadership(organization)
    results['social_presence'] = map_social_media(organization)
    
    # 3. Breach analysis
    results['organizational_breaches'] = search_org_breaches(organization)
    results['employee_breaches'] = search_employee_breaches(results['employees'])
    
    # 4. Threat landscape
    results['darknet_mentions'] = search_darknet_references(organization)
    results['threat_intel'] = gather_threat_intelligence(organization)
    
    return results
```

---

## Core Analysis Workflows

### Email and Breach Analysis

#### Comprehensive Email Investigation
```python
# Email analysis workflow example
def comprehensive_email_analysis(email_address):
    print(f"🔍 Analyzing email: {email_address}")
    
    # Step 1: Breach database search
    breach_results = invoke_breach_analyzer(email_address)
    print(f"📊 Found in {len(breach_results)} breaches")
    
    # Step 2: Social media correlation
    social_results = invoke_social_analyzer(email_address)
    print(f"📱 Found {len(social_results)} social profiles")
    
    # Step 3: Domain analysis
    domain = email_address.split('@')[1]
    domain_results = analyze_domain_infrastructure(domain)
    
    # Step 4: Pattern analysis
    similar_emails = find_similar_email_patterns(email_address)
    
    return {
        'target_email': email_address,
        'breach_data': breach_results,
        'social_profiles': social_results,
        'domain_analysis': domain_results,
        'related_emails': similar_emails,
        'risk_score': calculate_risk_score(breach_results),
        'analysis_timestamp': datetime.now().isoformat()
    }
```

#### Batch Email Analysis
```bash
# Batch email analysis via API
curl -X POST http://bev-api:8000/api/v1/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{
    "targets": [
      {"type": "email", "value": "target1@example.com"},
      {"type": "email", "value": "target2@example.com"},
      {"type": "email", "value": "target3@example.com"}
    ],
    "analysis_types": ["breach", "social", "domain"],
    "priority": "high"
  }'
```

### Cryptocurrency Analysis

#### Wallet Investigation Workflow
```python
def investigate_crypto_wallet(wallet_address, blockchain='bitcoin'):
    """
    Comprehensive cryptocurrency investigation
    """
    print(f"💰 Investigating {blockchain} wallet: {wallet_address}")
    
    # Step 1: Transaction history analysis
    transactions = get_transaction_history(wallet_address, blockchain)
    print(f"📈 Found {len(transactions)} transactions")
    
    # Step 2: Address clustering
    cluster_analysis = perform_address_clustering(wallet_address)
    related_addresses = cluster_analysis['related_addresses']
    
    # Step 3: Exchange interaction detection
    exchange_interactions = detect_exchange_interactions(wallet_address)
    
    # Step 4: Risk assessment
    risk_factors = assess_wallet_risk(wallet_address, transactions)
    
    # Step 5: Visualization data
    graph_data = build_transaction_graph(
        wallet_address, 
        related_addresses, 
        max_depth=3
    )
    
    return {
        'wallet_address': wallet_address,
        'blockchain': blockchain,
        'transaction_count': len(transactions),
        'total_value': sum(tx['value'] for tx in transactions),
        'related_addresses': related_addresses,
        'exchange_interactions': exchange_interactions,
        'risk_score': risk_factors['score'],
        'risk_factors': risk_factors['factors'],
        'graph_data': graph_data
    }
```

#### Cross-Blockchain Analysis
```python
# Multi-blockchain correlation
def cross_blockchain_analysis(identifier):
    """
    Analyze across multiple blockchains for correlation
    """
    blockchains = ['bitcoin', 'ethereum', 'litecoin', 'monero']
    results = {}
    
    for blockchain in blockchains:
        try:
            # Search for addresses associated with identifier
            addresses = find_addresses_for_identifier(identifier, blockchain)
            
            if addresses:
                results[blockchain] = {
                    'addresses': addresses,
                    'analysis': [investigate_crypto_wallet(addr, blockchain) 
                               for addr in addresses]
                }
        except Exception as e:
            print(f"⚠️ Error analyzing {blockchain}: {e}")
    
    # Cross-reference analysis
    correlations = find_cross_blockchain_correlations(results)
    
    return {
        'individual_analyses': results,
        'cross_correlations': correlations,
        'summary': generate_crypto_summary(results)
    }
```

### Social Media Intelligence

#### Comprehensive Social Profiling
```python
def comprehensive_social_analysis(target_identifier):
    """
    Multi-platform social media analysis
    """
    platforms = ['twitter', 'linkedin', 'facebook', 'instagram', 'reddit']
    results = {}
    
    for platform in platforms:
        try:
            profiles = search_platform_profiles(target_identifier, platform)
            
            for profile in profiles:
                analysis = {
                    'profile_data': extract_profile_data(profile),
                    'content_analysis': analyze_content_patterns(profile),
                    'network_analysis': map_social_network(profile),
                    'temporal_analysis': analyze_activity_patterns(profile),
                    'sentiment_analysis': analyze_content_sentiment(profile)
                }
                
                results[f"{platform}_{profile['id']}"] = analysis
                
        except Exception as e:
            print(f"⚠️ Error analyzing {platform}: {e}")
    
    # Cross-platform correlation
    correlations = find_social_correlations(results)
    
    return {
        'platform_analyses': results,
        'correlations': correlations,
        'timeline': build_social_timeline(results),
        'network_map': build_social_network_map(results)
    }
```

#### Social Network Mapping
```python
def map_social_network(primary_profile):
    """
    Map social connections and influence patterns
    """
    network = {
        'nodes': [{'id': primary_profile['id'], 'type': 'primary'}],
        'edges': [],
        'metrics': {}
    }
    
    # Level 1: Direct connections
    direct_connections = get_direct_connections(primary_profile)
    
    for connection in direct_connections:
        network['nodes'].append({
            'id': connection['id'],
            'type': 'direct',
            'relationship': connection['relationship_type'],
            'influence_score': calculate_influence_score(connection)
        })
        
        network['edges'].append({
            'source': primary_profile['id'],
            'target': connection['id'],
            'weight': connection['interaction_frequency']
        })
    
    # Level 2: Secondary connections (selective)
    high_influence_connections = [
        conn for conn in direct_connections 
        if conn['influence_score'] > 0.7
    ]
    
    for connection in high_influence_connections[:10]:  # Limit for performance
        secondary = get_direct_connections(connection)
        # Add secondary connections logic...
    
    # Calculate network metrics
    network['metrics'] = {
        'total_nodes': len(network['nodes']),
        'total_edges': len(network['edges']),
        'network_density': calculate_network_density(network),
        'centrality_measures': calculate_centrality(network),
        'community_detection': detect_communities(network)
    }
    
    return network
```

### Darknet Monitoring

#### Darknet Marketplace Analysis
```python
def analyze_darknet_presence(target_identifier):
    """
    Search darknet markets and forums for target presence
    """
    print("🕸️ Analyzing darknet presence (via Tor)")
    
    # Ensure Tor circuit is fresh
    rotate_tor_circuit()
    
    results = {
        'marketplaces': {},
        'forums': {},
        'mentions': {},
        'risk_assessment': {}
    }
    
    # Search major marketplaces
    marketplaces = ['alphabay', 'empire', 'darkmarket', 'whitehouse']
    
    for marketplace in marketplaces:
        try:
            # Search for vendor profiles
            vendor_profiles = search_marketplace_vendors(
                target_identifier, 
                marketplace
            )
            
            # Search for product listings
            product_mentions = search_marketplace_products(
                target_identifier,
                marketplace
            )
            
            results['marketplaces'][marketplace] = {
                'vendor_profiles': vendor_profiles,
                'product_mentions': product_mentions,
                'last_activity': get_last_activity_date(vendor_profiles)
            }
            
        except Exception as e:
            print(f"⚠️ Error searching {marketplace}: {e}")
    
    # Search forums and communities
    forums = ['dread', 'tor_carding', 'exploit_in']
    
    for forum in forums:
        try:
            forum_posts = search_forum_posts(target_identifier, forum)
            user_profiles = search_forum_users(target_identifier, forum)
            
            results['forums'][forum] = {
                'posts': forum_posts,
                'profiles': user_profiles,
                'reputation_score': calculate_reputation(user_profiles)
            }
            
        except Exception as e:
            print(f"⚠️ Error searching {forum}: {e}")
    
    # Risk assessment
    results['risk_assessment'] = assess_darknet_risk(results)
    
    return results
```

---

## Advanced Analysis Techniques

### Timeline Analysis

#### Activity Timeline Construction
```python
def build_comprehensive_timeline(investigation_data):
    """
    Build comprehensive timeline from all data sources
    """
    timeline_events = []
    
    # Extract events from different data sources
    
    # 1. Breach events
    for breach in investigation_data.get('breach_data', []):
        timeline_events.append({
            'timestamp': parse_date(breach['breach_date']),
            'event_type': 'data_breach',
            'description': f"Data breach at {breach['company']}",
            'source': 'breach_database',
            'confidence': 0.9,
            'details': breach
        })
    
    # 2. Social media activity
    for platform, data in investigation_data.get('social_profiles', {}).items():
        for post in data.get('posts', []):
            timeline_events.append({
                'timestamp': parse_date(post['date']),
                'event_type': 'social_activity',
                'description': f"Post on {platform}: {post['content'][:50]}...",
                'source': platform,
                'confidence': 0.8,
                'details': post
            })
    
    # 3. Cryptocurrency transactions
    for tx in investigation_data.get('crypto_transactions', []):
        timeline_events.append({
            'timestamp': parse_date(tx['timestamp']),
            'event_type': 'crypto_transaction',
            'description': f"Transaction: {tx['value']} {tx['currency']}",
            'source': 'blockchain',
            'confidence': 1.0,
            'details': tx
        })
    
    # 4. Darknet activity
    for activity in investigation_data.get('darknet_activity', []):
        timeline_events.append({
            'timestamp': parse_date(activity['date']),
            'event_type': 'darknet_activity',
            'description': f"Activity on {activity['platform']}",
            'source': 'darknet_monitoring',
            'confidence': 0.7,
            'details': activity
        })
    
    # Sort by timestamp
    timeline_events.sort(key=lambda x: x['timestamp'])
    
    # Identify patterns and clusters
    patterns = identify_temporal_patterns(timeline_events)
    clusters = cluster_temporal_events(timeline_events)
    
    return {
        'events': timeline_events,
        'patterns': patterns,
        'clusters': clusters,
        'summary': generate_timeline_summary(timeline_events)
    }
```

### Attribution Analysis

#### Multi-Factor Attribution Scoring
```python
def calculate_attribution_confidence(evidence_points):
    """
    Calculate attribution confidence based on multiple evidence factors
    """
    factors = {
        'technical_indicators': 0.0,
        'behavioral_patterns': 0.0,
        'temporal_correlations': 0.0,
        'social_connections': 0.0,
        'financial_links': 0.0
    }
    
    # Technical indicators (40% weight)
    tech_score = 0
    if evidence_points.get('email_matches'):
        tech_score += 0.3
    if evidence_points.get('username_patterns'):
        tech_score += 0.2
    if evidence_points.get('device_fingerprints'):
        tech_score += 0.3
    if evidence_points.get('ip_geolocation'):
        tech_score += 0.2
    factors['technical_indicators'] = min(tech_score, 1.0)
    
    # Behavioral patterns (25% weight)
    behavioral_score = 0
    if evidence_points.get('activity_patterns'):
        behavioral_score += 0.4
    if evidence_points.get('communication_style'):
        behavioral_score += 0.3
    if evidence_points.get('operational_security'):
        behavioral_score += 0.3
    factors['behavioral_patterns'] = min(behavioral_score, 1.0)
    
    # Temporal correlations (15% weight)
    temporal_score = 0
    if evidence_points.get('timeline_matches'):
        temporal_score += 0.6
    if evidence_points.get('activity_overlap'):
        temporal_score += 0.4
    factors['temporal_correlations'] = min(temporal_score, 1.0)
    
    # Social connections (10% weight)
    social_score = 0
    if evidence_points.get('shared_contacts'):
        social_score += 0.5
    if evidence_points.get('social_validation'):
        social_score += 0.5
    factors['social_connections'] = min(social_score, 1.0)
    
    # Financial links (10% weight)
    financial_score = 0
    if evidence_points.get('shared_accounts'):
        financial_score += 0.6
    if evidence_points.get('transaction_patterns'):
        financial_score += 0.4
    factors['financial_links'] = min(financial_score, 1.0)
    
    # Calculate weighted confidence score
    weights = {
        'technical_indicators': 0.40,
        'behavioral_patterns': 0.25,
        'temporal_correlations': 0.15,
        'social_connections': 0.10,
        'financial_links': 0.10
    }
    
    confidence_score = sum(
        factors[factor] * weights[factor] 
        for factor in factors
    )
    
    # Confidence categories
    if confidence_score >= 0.8:
        confidence_level = "HIGH"
    elif confidence_score >= 0.6:
        confidence_level = "MEDIUM"
    elif confidence_score >= 0.4:
        confidence_level = "LOW"
    else:
        confidence_level = "INSUFFICIENT"
    
    return {
        'overall_score': confidence_score,
        'confidence_level': confidence_level,
        'factor_scores': factors,
        'weights_applied': weights,
        'evidence_summary': evidence_points
    }
```

---

## Data Correlation & Visualization

### Graph-Based Analysis

#### Relationship Network Visualization
```python
def create_investigation_graph(investigation_data):
    """
    Create comprehensive graph visualization of investigation data
    """
    # Initialize graph with Cytoscape.js format
    graph = {
        'nodes': [],
        'edges': [],
        'layout': 'force-directed',
        'style': 'dark_theme'
    }
    
    # Add nodes for different entity types
    entities = extract_entities(investigation_data)
    
    for entity in entities:
        node = {
            'data': {
                'id': entity['id'],
                'label': entity['name'],
                'type': entity['type'],
                'confidence': entity.get('confidence', 0.5),
                'attributes': entity.get('attributes', {})
            },
            'classes': f"entity_{entity['type']}"
        }
        
        # Add risk scoring for visual emphasis
        if entity['type'] == 'email':
            risk_score = calculate_email_risk(entity)
            node['data']['risk'] = risk_score
            node['classes'] += f" risk_{get_risk_category(risk_score)}"
        
        graph['nodes'].append(node)
    
    # Add edges for relationships
    relationships = extract_relationships(investigation_data)
    
    for relationship in relationships:
        edge = {
            'data': {
                'id': f"{relationship['source']}->{relationship['target']}",
                'source': relationship['source'],
                'target': relationship['target'],
                'relationship_type': relationship['type'],
                'confidence': relationship.get('confidence', 0.5),
                'weight': relationship.get('weight', 1)
            },
            'classes': f"relationship_{relationship['type']}"
        }
        
        graph['edges'].append(edge)
    
    return graph
```

#### Advanced Graph Analytics
```python
def perform_graph_analysis(graph_data):
    """
    Perform advanced graph analytics on investigation data
    """
    import networkx as nx
    
    # Convert to NetworkX graph for analysis
    G = nx.Graph()
    
    # Add nodes with attributes
    for node in graph_data['nodes']:
        G.add_node(
            node['data']['id'],
            **node['data']
        )
    
    # Add edges with weights
    for edge in graph_data['edges']:
        G.add_edge(
            edge['data']['source'],
            edge['data']['target'],
            weight=edge['data']['weight'],
            **edge['data']
        )
    
    # Calculate graph metrics
    metrics = {
        'centrality_measures': {
            'betweenness': nx.betweenness_centrality(G),
            'closeness': nx.closeness_centrality(G),
            'degree': nx.degree_centrality(G),
            'eigenvector': nx.eigenvector_centrality(G)
        },
        'community_detection': detect_communities(G),
        'shortest_paths': calculate_key_paths(G),
        'clustering_coefficient': nx.clustering(G),
        'graph_statistics': {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'diameter': nx.diameter(G) if nx.is_connected(G) else None
        }
    }
    
    # Identify key nodes and paths
    key_insights = {
        'most_central_nodes': get_top_central_nodes(metrics['centrality_measures']),
        'bridge_nodes': identify_bridge_nodes(G),
        'isolated_components': list(nx.connected_components(G)),
        'potential_pivots': identify_pivot_nodes(G, metrics)
    }
    
    return {
        'metrics': metrics,
        'insights': key_insights,
        'recommendations': generate_analysis_recommendations(metrics, key_insights)
    }
```

### Timeline Visualization

#### Interactive Timeline Creation
```javascript
// Frontend timeline visualization
function createInvestigationTimeline(timelineData) {
    const timeline = {
        items: [],
        groups: [],
        options: {
            start: new Date(timelineData.start_date),
            end: new Date(timelineData.end_date),
            height: '600px',
            theme: 'dark',
            zoomable: true,
            moveable: true
        }
    };
    
    // Create groups for different data sources
    const sources = [...new Set(timelineData.events.map(e => e.source))];
    sources.forEach((source, index) => {
        timeline.groups.push({
            id: index,
            content: source.toUpperCase(),
            className: `timeline-group-${source}`
        });
    });
    
    // Add events to timeline
    timelineData.events.forEach((event, index) => {
        const groupId = sources.indexOf(event.source);
        
        timeline.items.push({
            id: index,
            group: groupId,
            start: new Date(event.timestamp),
            content: event.description,
            title: `${event.event_type}: ${event.description}`,
            className: `timeline-item-${event.event_type}`,
            style: `background-color: ${getEventColor(event.event_type)}`
        });
    });
    
    return timeline;
}

// Event color coding
function getEventColor(eventType) {
    const colors = {
        'data_breach': '#e74c3c',
        'social_activity': '#3498db',
        'crypto_transaction': '#f39c12',
        'darknet_activity': '#9b59b6',
        'email_activity': '#2ecc71'
    };
    return colors[eventType] || '#95a5a6';
}
```

---

## Operational Security for Analysts

### Investigation OpSec Procedures

#### Pre-Investigation Checklist
```bash
#!/bin/bash
# Analyst OpSec verification script

echo "🔒 ANALYST OPSEC CHECKLIST"
echo "=========================="

# 1. Verify Tor connectivity
check_tor_connection() {
    echo "🌐 Checking Tor connectivity..."
    
    # Test Tor proxy
    tor_result=$(curl -s -x socks5://127.0.0.1:9050 \
        http://check.torproject.org/ | grep -o "Congratulations")
    
    if [ "$tor_result" == "Congratulations" ]; then
        echo "✅ Tor proxy working"
    else
        echo "❌ Tor proxy not working"
        return 1
    fi
    
    # Check circuit status
    circuit_count=$(ss -tuln | grep ":9050" | wc -l)
    echo "🔄 Active Tor circuits: $circuit_count"
}

# 2. Verify VPN status (if applicable)
check_vpn_status() {
    echo "🛡️ Checking VPN status..."
    
    # Check for VPN interfaces
    vpn_interfaces=$(ip link | grep -E "(tun|tap)" | wc -l)
    
    if [ $vpn_interfaces -gt 0 ]; then
        echo "✅ VPN interface detected"
    else
        echo "⚠️ No VPN interface found"
    fi
    
    # Check public IP (should be VPN IP, not real IP)
    public_ip=$(curl -s -x socks5://127.0.0.1:9050 http://ifconfig.me)
    echo "🌍 Public IP: $public_ip"
}

# 3. Verify browser security
check_browser_security() {
    echo "🌐 Browser security recommendations:"
    echo "  - [ ] Using private/incognito mode"
    echo "  - [ ] JavaScript disabled by default"
    echo "  - [ ] Third-party cookies blocked"
    echo "  - [ ] Location services disabled"
    echo "  - [ ] Auto-downloads disabled"
}

# 4. Check system security
check_system_security() {
    echo "🖥️ System security status..."
    
    # Check for active SSH connections
    ssh_connections=$(netstat -tn | grep :22 | grep ESTABLISHED | wc -l)
    echo "🔐 Active SSH connections: $ssh_connections"
    
    # Check disk encryption status
    if [ -f /sys/fs/ext4/*/encryption/policy ]; then
        echo "✅ Disk encryption detected"
    else
        echo "⚠️ Disk encryption not detected"
    fi
}

# Run all checks
check_tor_connection
check_vpn_status
check_browser_security
check_system_security

echo ""
echo "🎯 Ready for secure investigation"
```

#### During Investigation Procedures
```python
# Investigation OpSec monitoring
class InvestigationOpSec:
    def __init__(self):
        self.session_start = datetime.now()
        self.circuit_rotations = 0
        self.last_rotation = datetime.now()
        
    def monitor_session(self):
        """Monitor investigation session for OpSec compliance"""
        
        # Rotate Tor circuit every 10 minutes
        time_since_rotation = datetime.now() - self.last_rotation
        if time_since_rotation > timedelta(minutes=10):
            self.rotate_tor_circuit()
            self.last_rotation = datetime.now()
            self.circuit_rotations += 1
        
        # Check for long sessions (> 2 hours)
        session_duration = datetime.now() - self.session_start
        if session_duration > timedelta(hours=2):
            self.session_warning()
        
        # Monitor request patterns
        self.check_request_patterns()
    
    def rotate_tor_circuit(self):
        """Rotate Tor circuit for fresh identity"""
        try:
            with stem.control.Controller.from_port(port=9051) as controller:
                controller.authenticate()
                controller.signal(stem.Signal.NEWNYM)
                time.sleep(10)  # Wait for new circuit
                print("🔄 Tor circuit rotated")
        except Exception as e:
            print(f"❌ Circuit rotation failed: {e}")
    
    def session_warning(self):
        """Warning for extended investigation sessions"""
        print("⚠️ Investigation session > 2 hours")
        print("   Consider taking a break for OpSec")
        print("   Recommended: Rotate circuits and clear caches")
    
    def check_request_patterns(self):
        """Monitor for suspicious request patterns"""
        # Check request frequency
        # Monitor for rate limiting responses
        # Validate anonymity chain integrity
        pass
```

### Data Handling Procedures

#### Evidence Collection Standards
```python
def collect_evidence(data, source, confidence_level):
    """
    Collect evidence with proper chain of custody
    """
    evidence_record = {
        'id': generate_evidence_id(),
        'timestamp': datetime.now().isoformat(),
        'source': source,
        'confidence_level': confidence_level,
        'data_hash': calculate_hash(data),
        'collection_method': get_collection_method(source),
        'analyst_id': get_current_analyst(),
        'preservation_method': 'digital_copy',
        'chain_of_custody': []
    }
    
    # Add initial chain of custody entry
    evidence_record['chain_of_custody'].append({
        'action': 'collected',
        'timestamp': datetime.now().isoformat(),
        'analyst': get_current_analyst(),
        'method': evidence_record['collection_method']
    })
    
    # Store evidence with encryption
    stored_path = store_evidence_securely(data, evidence_record)
    evidence_record['storage_location'] = stored_path
    
    # Update evidence database
    update_evidence_database(evidence_record)
    
    return evidence_record

def verify_evidence_integrity(evidence_id):
    """
    Verify evidence hasn't been tampered with
    """
    evidence_record = get_evidence_record(evidence_id)
    current_data = retrieve_evidence_data(evidence_record['storage_location'])
    current_hash = calculate_hash(current_data)
    
    if current_hash == evidence_record['data_hash']:
        return True, "Evidence integrity verified"
    else:
        return False, "Evidence integrity compromised"
```

---

## Report Generation & Documentation

### Structured Report Templates

#### Executive Summary Template
```markdown
# OSINT Investigation Report
## Executive Summary

**Investigation ID**: {investigation_id}
**Target(s)**: {target_list}
**Investigation Period**: {start_date} - {end_date}
**Analyst**: {analyst_name}
**Classification**: {classification_level}

### Key Findings Summary
- **Breach Exposure**: {breach_count} data breaches identified
- **Financial Links**: {crypto_analysis_summary}
- **Social Presence**: {social_media_summary}
- **Risk Assessment**: {overall_risk_level}
- **Attribution Confidence**: {attribution_confidence}

### Critical Intelligence
{critical_findings}

### Recommendations
{actionable_recommendations}

### Executive Timeline
{high_level_timeline}
```

#### Technical Report Template
```markdown
# Technical Analysis Report

## Methodology
**Analysis Framework**: {methodology_used}
**Tools Utilized**: {tools_list}
**Data Sources**: {sources_accessed}
**Analysis Period**: {analysis_timeframe}

## Detailed Findings

### Breach Database Analysis
**Sources Checked**: {breach_sources}
**Exposures Found**: {breach_details}
**Risk Score**: {breach_risk_score}

### Social Media Intelligence
**Platforms Analyzed**: {platforms_list}
**Profiles Identified**: {profile_count}
**Network Analysis**: {network_summary}

### Cryptocurrency Analysis
**Wallets Analyzed**: {wallet_count}
**Transaction Volume**: {transaction_summary}
**Exchange Interactions**: {exchange_analysis}

### Darknet Monitoring
**Markets Searched**: {markets_accessed}
**Forums Monitored**: {forums_checked}
**Mentions Found**: {mention_count}

## Attribution Analysis
**Confidence Score**: {attribution_score}
**Supporting Evidence**: {evidence_summary}
**Correlation Analysis**: {correlation_results}

## Technical Appendix
{technical_details}
```

### Automated Report Generation

#### Report Generation API
```python
class ReportGenerator:
    def __init__(self, investigation_data):
        self.data = investigation_data
        self.templates = load_report_templates()
    
    def generate_executive_summary(self):
        """Generate executive-level summary report"""
        template = self.templates['executive_summary']
        
        summary_data = {
            'investigation_id': self.data['investigation_id'],
            'target_list': ', '.join(self.data['targets']),
            'start_date': self.data['start_date'],
            'end_date': self.data['end_date'],
            'analyst_name': self.data['analyst'],
            'classification_level': self.data['classification'],
            'breach_count': len(self.data.get('breach_results', [])),
            'crypto_analysis_summary': self.summarize_crypto_analysis(),
            'social_media_summary': self.summarize_social_analysis(),
            'overall_risk_level': self.calculate_overall_risk(),
            'attribution_confidence': self.data.get('attribution_confidence', 'N/A'),
            'critical_findings': self.extract_critical_findings(),
            'actionable_recommendations': self.generate_recommendations(),
            'high_level_timeline': self.create_executive_timeline()
        }
        
        return template.format(**summary_data)
    
    def generate_technical_report(self):
        """Generate detailed technical report"""
        template = self.templates['technical_report']
        
        technical_data = {
            'methodology_used': self.data['methodology'],
            'tools_list': ', '.join(self.data['tools_used']),
            'sources_accessed': self.list_data_sources(),
            'analysis_timeframe': f"{self.data['start_date']} - {self.data['end_date']}",
            'breach_sources': self.list_breach_sources(),
            'breach_details': self.format_breach_details(),
            'breach_risk_score': self.calculate_breach_risk(),
            'platforms_list': self.list_social_platforms(),
            'profile_count': self.count_social_profiles(),
            'network_summary': self.summarize_social_network(),
            'wallet_count': self.count_crypto_wallets(),
            'transaction_summary': self.summarize_transactions(),
            'exchange_analysis': self.analyze_exchange_interactions(),
            'markets_accessed': self.list_darknet_markets(),
            'forums_checked': self.list_darknet_forums(),
            'mention_count': self.count_darknet_mentions(),
            'attribution_score': self.data.get('attribution_score', 'N/A'),
            'evidence_summary': self.summarize_evidence(),
            'correlation_results': self.summarize_correlations(),
            'technical_details': self.format_technical_appendix()
        }
        
        return template.format(**technical_data)
    
    def generate_evidence_package(self):
        """Generate evidence package for legal/compliance use"""
        evidence_package = {
            'metadata': {
                'package_id': generate_package_id(),
                'creation_date': datetime.now().isoformat(),
                'analyst': self.data['analyst'],
                'case_id': self.data['investigation_id']
            },
            'evidence_items': [],
            'chain_of_custody': [],
            'verification_hashes': {}
        }
        
        # Collect all evidence items
        for evidence_type in ['breach_data', 'social_profiles', 'crypto_analysis', 'darknet_findings']:
            if evidence_type in self.data:
                evidence_items = self.package_evidence(evidence_type, self.data[evidence_type])
                evidence_package['evidence_items'].extend(evidence_items)
        
        # Generate verification hashes
        for item in evidence_package['evidence_items']:
            item_hash = calculate_hash(item['data'])
            evidence_package['verification_hashes'][item['id']] = item_hash
        
        return evidence_package
```

---

## Case Studies & Examples

### Case Study 1: Email Breach Investigation

#### Scenario
```
Target: suspect@example.com
Objective: Determine breach exposure and associated risks
Time Frame: 2 hours
Classification: CONFIDENTIAL
```

#### Investigation Workflow
```python
# Example investigation execution
def investigate_email_breach_case():
    target_email = "suspect@example.com"
    
    print(f"🎯 Starting investigation: {target_email}")
    
    # Phase 1: Initial collection
    print("📊 Phase 1: Data Collection")
    breach_results = invoke_breach_analyzer(target_email)
    social_results = invoke_social_analyzer(target_email)
    
    print(f"   Found in {len(breach_results)} breaches")
    print(f"   Found {len(social_results)} social profiles")
    
    # Phase 2: Enrichment
    print("🔍 Phase 2: Data Enrichment")
    domain = target_email.split('@')[1]
    domain_analysis = analyze_domain_infrastructure(domain)
    
    # Look for similar email patterns
    similar_emails = find_similar_email_patterns(target_email)
    
    # Phase 3: Correlation
    print("🔗 Phase 3: Correlation Analysis")
    timeline = build_activity_timeline({
        'breach_data': breach_results,
        'social_profiles': social_results
    })
    
    # Phase 4: Analysis
    print("📈 Phase 4: Risk Analysis")
    risk_score = calculate_comprehensive_risk({
        'breach_count': len(breach_results),
        'social_exposure': len(social_results),
        'domain_reputation': domain_analysis['reputation_score'],
        'timeline_patterns': timeline['risk_indicators']
    })
    
    # Generate findings summary
    findings = {
        'target_email': target_email,
        'breach_exposure': {
            'total_breaches': len(breach_results),
            'high_risk_breaches': [b for b in breach_results if b['severity'] == 'high'],
            'most_recent': max(breach_results, key=lambda x: x['date'])['date']
        },
        'social_presence': {
            'platforms_found': list(social_results.keys()),
            'total_profiles': len(social_results),
            'activity_level': calculate_activity_level(social_results)
        },
        'risk_assessment': {
            'overall_score': risk_score,
            'risk_level': get_risk_category(risk_score),
            'primary_risks': identify_primary_risks(breach_results, social_results)
        },
        'recommendations': generate_risk_recommendations(risk_score, breach_results)
    }
    
    return findings
```

#### Key Findings Example
```json
{
  "target_email": "suspect@example.com",
  "breach_exposure": {
    "total_breaches": 7,
    "high_risk_breaches": [
      {
        "company": "LinkedInBreach2021",
        "date": "2021-06-15",
        "exposed_data": ["email", "password_hash", "profile_data"],
        "severity": "high"
      }
    ],
    "most_recent": "2023-11-20"
  },
  "social_presence": {
    "platforms_found": ["twitter", "linkedin", "reddit"],
    "total_profiles": 3,
    "activity_level": "moderate"
  },
  "risk_assessment": {
    "overall_score": 0.78,
    "risk_level": "HIGH",
    "primary_risks": [
      "Credential reuse across platforms",
      "Personal information exposure",
      "Social engineering vulnerability"
    ]
  },
  "recommendations": [
    "Immediate password reset recommended",
    "Enable 2FA on all associated accounts",
    "Monitor for identity theft indicators",
    "Consider credit monitoring services"
  ]
}
```

### Case Study 2: Cryptocurrency Investigation

#### Scenario
```
Target: Bitcoin wallet address (1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa)
Objective: Trace transaction history and identify connections
Time Frame: 4 hours
Classification: RESTRICTED
```

#### Investigation Results Summary
```python
# Example cryptocurrency investigation results
crypto_investigation_results = {
    "wallet_address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
    "blockchain": "bitcoin",
    "analysis_summary": {
        "total_transactions": 2847,
        "total_volume": "184.32 BTC",
        "first_activity": "2009-01-03",
        "last_activity": "2024-03-15",
        "exchange_interactions": 12,
        "risk_score": 0.23
    },
    "key_findings": [
        "Historic wallet with early Bitcoin activity",
        "Connections to multiple exchanges",
        "Pattern suggesting institutional usage",
        "No direct criminal associations found"
    ],
    "related_addresses": [
        "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2",
        "3FupnqLpnUYF7o3EQ8KrBxk6QhfzKVjqoW"
    ],
    "attribution_confidence": {
        "score": 0.45,
        "level": "LOW",
        "reasoning": "Limited identifying information available"
    }
}
```

### Case Study 3: Social Media Network Analysis

#### Scenario
```
Target: Social media profile @suspect_user
Objective: Map social network and identify key connections
Time Frame: 3 hours
Classification: CONFIDENTIAL
```

#### Network Analysis Results
```json
{
  "target_profile": "@suspect_user",
  "platform": "twitter",
  "network_analysis": {
    "direct_connections": 247,
    "high_influence_connections": 12,
    "community_clusters": 3,
    "network_centrality": 0.67
  },
  "key_influencers": [
    {
      "username": "@key_contact_1",
      "influence_score": 0.89,
      "relationship_type": "frequent_interaction",
      "shared_interests": ["cybersecurity", "OSINT"]
    }
  ],
  "behavioral_patterns": {
    "posting_frequency": "3.2 posts/day",
    "peak_activity_hours": ["09:00-11:00", "20:00-22:00"],
    "sentiment_trend": "neutral_to_positive",
    "topic_focus": ["technology", "security", "politics"]
  },
  "risk_indicators": [
    "Frequent interaction with known threat actors",
    "Sharing of sensitive information",
    "Potential sockpuppet accounts identified"
  ]
}
```

---

*Last Updated: 2025-09-19*
*Framework Version: BEV OSINT v2.0*
*Classification: INTERNAL*
*Document Version: 1.0*