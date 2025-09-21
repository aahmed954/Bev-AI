export { matchers } from './matchers.js';

export const nodes = [
	() => import('./nodes/0'),
	() => import('./nodes/1'),
	() => import('./nodes/2'),
	() => import('./nodes/3'),
	() => import('./nodes/4'),
	() => import('./nodes/5'),
	() => import('./nodes/6'),
	() => import('./nodes/7'),
	() => import('./nodes/8'),
	() => import('./nodes/9'),
	() => import('./nodes/10'),
	() => import('./nodes/11'),
	() => import('./nodes/12'),
	() => import('./nodes/13'),
	() => import('./nodes/14'),
	() => import('./nodes/15'),
	() => import('./nodes/16'),
	() => import('./nodes/17'),
	() => import('./nodes/18'),
	() => import('./nodes/19'),
	() => import('./nodes/20'),
	() => import('./nodes/21'),
	() => import('./nodes/22'),
	() => import('./nodes/23'),
	() => import('./nodes/24'),
	() => import('./nodes/25'),
	() => import('./nodes/26'),
	() => import('./nodes/27'),
	() => import('./nodes/28'),
	() => import('./nodes/29'),
	() => import('./nodes/30'),
	() => import('./nodes/31'),
	() => import('./nodes/32'),
	() => import('./nodes/33'),
	() => import('./nodes/34'),
	() => import('./nodes/35'),
	() => import('./nodes/36'),
	() => import('./nodes/37'),
	() => import('./nodes/38'),
	() => import('./nodes/39'),
	() => import('./nodes/40'),
	() => import('./nodes/41'),
	() => import('./nodes/42'),
	() => import('./nodes/43'),
	() => import('./nodes/44'),
	() => import('./nodes/45'),
	() => import('./nodes/46'),
	() => import('./nodes/47'),
	() => import('./nodes/48')
];

export const server_loads = [];

export const dictionary = {
		"/": [2],
		"/ai-pipeline": [3],
		"/analyzers": [4],
		"/autonomous": [5],
		"/avatar": [6],
		"/chaos/engineering": [7],
		"/config": [8],
		"/containers": [9],
		"/crypto": [10],
		"/darknet": [11],
		"/databases/vector": [13],
		"/database": [12],
		"/deployment": [14],
		"/devops": [15],
		"/edge": [16],
		"/enhancement": [17],
		"/extended-reasoning": [18],
		"/infrastructure": [19],
		"/infrastructure/docker": [20],
		"/intelowl-admin": [21],
		"/knowledge": [22],
		"/market-intel": [23],
		"/mcp-admin": [24],
		"/message-queues": [25],
		"/ml-pipeline": [26],
		"/monitoring/alerts": [27],
		"/monitoring/logs": [28],
		"/multimodal": [29],
		"/ocr": [30],
		"/oracle": [31],
		"/performance": [32],
		"/phase9/api": [33],
		"/phase9/autonomous": [34],
		"/phase9/research": [35],
		"/phase9/swarm": [36],
		"/pipelines/airflow": [37],
		"/recovery": [38],
		"/research": [39],
		"/security-ops": [41],
		"/security/soc": [40],
		"/social-media": [42],
		"/swarm-master": [43],
		"/testing": [44],
		"/threat-intel": [45],
		"/tor": [46],
		"/visualization": [47],
		"/workflows/n8n": [48]
	};

export const hooks = {
	handleError: (({ error }) => { console.error(error) }),
	
	reroute: (() => {}),
	transport: {}
};

export const decoders = Object.fromEntries(Object.entries(hooks.transport).map(([k, v]) => [k, v.decode]));

export const hash = false;

export const decode = (type, value) => decoders[type](value);

export { default as root } from '../root.svelte';