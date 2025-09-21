
// this file is generated — do not edit it


declare module "svelte/elements" {
	export interface HTMLAttributes<T> {
		'data-sveltekit-keepfocus'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-noscroll'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-preload-code'?:
			| true
			| ''
			| 'eager'
			| 'viewport'
			| 'hover'
			| 'tap'
			| 'off'
			| undefined
			| null;
		'data-sveltekit-preload-data'?: true | '' | 'hover' | 'tap' | 'off' | undefined | null;
		'data-sveltekit-reload'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-replacestate'?: true | '' | 'off' | undefined | null;
	}
}

export {};


declare module "$app/types" {
	export interface AppTypes {
		RouteId(): "/" | "/analyzers" | "/analyzers/social" | "/crypto" | "/darknet" | "/database" | "/edge" | "/infrastructure" | "/knowledge" | "/logs" | "/ml-pipeline" | "/ocr" | "/performance" | "/security-ops" | "/threat-intel";
		RouteParams(): {
			
		};
		LayoutParams(): {
			"/": Record<string, never>;
			"/analyzers": Record<string, never>;
			"/analyzers/social": Record<string, never>;
			"/crypto": Record<string, never>;
			"/darknet": Record<string, never>;
			"/database": Record<string, never>;
			"/edge": Record<string, never>;
			"/infrastructure": Record<string, never>;
			"/knowledge": Record<string, never>;
			"/logs": Record<string, never>;
			"/ml-pipeline": Record<string, never>;
			"/ocr": Record<string, never>;
			"/performance": Record<string, never>;
			"/security-ops": Record<string, never>;
			"/threat-intel": Record<string, never>
		};
		Pathname(): "/" | "/analyzers" | "/analyzers/" | "/analyzers/social" | "/analyzers/social/" | "/crypto" | "/crypto/" | "/darknet" | "/darknet/" | "/database" | "/database/" | "/edge" | "/edge/" | "/infrastructure" | "/infrastructure/" | "/knowledge" | "/knowledge/" | "/logs" | "/logs/" | "/ml-pipeline" | "/ml-pipeline/" | "/ocr" | "/ocr/" | "/performance" | "/performance/" | "/security-ops" | "/security-ops/" | "/threat-intel" | "/threat-intel/";
		ResolvedPathname(): `${"" | `/${string}`}${ReturnType<AppTypes['Pathname']>}`;
		Asset(): string & {};
	}
}