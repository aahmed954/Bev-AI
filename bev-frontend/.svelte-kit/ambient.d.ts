
// this file is generated — do not edit it


/// <reference types="@sveltejs/kit" />

/**
 * Environment variables [loaded by Vite](https://vitejs.dev/guide/env-and-mode.html#env-files) from `.env` files and `process.env`. Like [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), this module cannot be imported into client-side code. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured).
 * 
 * _Unlike_ [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), the values exported from this module are statically injected into your bundle at build time, enabling optimisations like dead code elimination.
 * 
 * ```ts
 * import { API_KEY } from '$env/static/private';
 * ```
 * 
 * Note that all environment variables referenced in your code should be declared (for example in an `.env` file), even if they don't have a value until the app is deployed:
 * 
 * ```
 * MY_FEATURE_FLAG=""
 * ```
 * 
 * You can override `.env` values from the command line like so:
 * 
 * ```sh
 * MY_FEATURE_FLAG="enabled" npm run dev
 * ```
 */
declare module '$env/static/private' {
	export const GITHUB_PAT: string;
	export const USER: string;
	export const CLAUDE_CODE_ENTRYPOINT: string;
	export const npm_config_user_agent: string;
	export const GIT_EDITOR: string;
	export const XDG_SESSION_TYPE: string;
	export const BUN_INSTALL: string;
	export const npm_node_execpath: string;
	export const XDG_CACHE_HOME: string;
	export const SHLVL: string;
	export const npm_config_noproxy: string;
	export const OLDPWD: string;
	export const HOME: string;
	export const LESS: string;
	export const DESKTOP_SESSION: string;
	export const NVM_BIN: string;
	export const npm_package_json: string;
	export const ZSH: string;
	export const LSCOLORS: string;
	export const NVM_INC: string;
	export const GTK_MODULES: string;
	export const HF_HOME: string;
	export const GNOME_SHELL_SESSION_MODE: string;
	export const PAGER: string;
	export const npm_config_userconfig: string;
	export const npm_config_local_prefix: string;
	export const SYSTEMD_EXEC_PID: string;
	export const DBUS_SESSION_BUS_ADDRESS: string;
	export const GSM_SKIP_SSH_AGENT_WORKAROUND: string;
	export const DBUS_STARTER_BUS_TYPE: string;
	export const COLORTERM: string;
	export const COLOR: string;
	export const NVM_DIR: string;
	export const WAYLAND_DISPLAY: string;
	export const LOGNAME: string;
	export const _: string;
	export const PIP_USER: string;
	export const npm_config_prefix: string;
	export const npm_config_npm_version: string;
	export const XDG_SESSION_CLASS: string;
	export const MEMORY_PRESSURE_WATCH: string;
	export const USERNAME: string;
	export const TERM: string;
	export const OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE: string;
	export const npm_config_cache: string;
	export const GNOME_DESKTOP_SESSION_ID: string;
	export const RUSTUP_HOME: string;
	export const npm_config_node_gyp: string;
	export const PATH: string;
	export const SESSION_MANAGER: string;
	export const NODE: string;
	export const npm_package_name: string;
	export const GNOME_TERMINAL_SCREEN: string;
	export const TRANSFORMERS_CACHE: string;
	export const XDG_MENU_PREFIX: string;
	export const GNOME_SETUP_DISPLAY: string;
	export const XDG_RUNTIME_DIR: string;
	export const COREPACK_ENABLE_AUTO_PIN: string;
	export const DISPLAY: string;
	export const LANG: string;
	export const XDG_CURRENT_DESKTOP: string;
	export const PYTHONUSERBASE: string;
	export const NoDefaultCurrentDirectoryInExePath: string;
	export const XDG_SESSION_DESKTOP: string;
	export const GNOME_TERMINAL_SERVICE: string;
	export const XMODIFIERS: string;
	export const XAUTHORITY: string;
	export const LS_COLORS: string;
	export const npm_lifecycle_script: string;
	export const SSH_AUTH_SOCK: string;
	export const SHELL: string;
	export const GOPATH: string;
	export const npm_package_version: string;
	export const npm_lifecycle_event: string;
	export const QT_ACCESSIBILITY: string;
	export const GDMSESSION: string;
	export const CLAUDECODE: string;
	export const QT_IM_MODULE: string;
	export const npm_config_globalconfig: string;
	export const npm_config_init_module: string;
	export const PWD: string;
	export const npm_execpath: string;
	export const XDG_DATA_DIRS: string;
	export const DBUS_STARTER_ADDRESS: string;
	export const CARGO_HOME: string;
	export const NVM_CD_FLAGS: string;
	export const npm_config_global_prefix: string;
	export const npm_command: string;
	export const CONDA_PREFIX: string;
	export const VTE_VERSION: string;
	export const MEMORY_PRESSURE_WRITE: string;
	export const INIT_CWD: string;
	export const EDITOR: string;
	export const NODE_ENV: string;
}

/**
 * Similar to [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private), except that it only includes environment variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Values are replaced statically at build time.
 * 
 * ```ts
 * import { PUBLIC_BASE_URL } from '$env/static/public';
 * ```
 */
declare module '$env/static/public' {
	
}

/**
 * This module provides access to runtime environment variables, as defined by the platform you're running on. For example if you're using [`adapter-node`](https://github.com/sveltejs/kit/tree/main/packages/adapter-node) (or running [`vite preview`](https://svelte.dev/docs/kit/cli)), this is equivalent to `process.env`. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured).
 * 
 * This module cannot be imported into client-side code.
 * 
 * ```ts
 * import { env } from '$env/dynamic/private';
 * console.log(env.DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 * 
 * > [!NOTE] In `dev`, `$env/dynamic` always includes environment variables from `.env`. In `prod`, this behavior will depend on your adapter.
 */
declare module '$env/dynamic/private' {
	export const env: {
		GITHUB_PAT: string;
		USER: string;
		CLAUDE_CODE_ENTRYPOINT: string;
		npm_config_user_agent: string;
		GIT_EDITOR: string;
		XDG_SESSION_TYPE: string;
		BUN_INSTALL: string;
		npm_node_execpath: string;
		XDG_CACHE_HOME: string;
		SHLVL: string;
		npm_config_noproxy: string;
		OLDPWD: string;
		HOME: string;
		LESS: string;
		DESKTOP_SESSION: string;
		NVM_BIN: string;
		npm_package_json: string;
		ZSH: string;
		LSCOLORS: string;
		NVM_INC: string;
		GTK_MODULES: string;
		HF_HOME: string;
		GNOME_SHELL_SESSION_MODE: string;
		PAGER: string;
		npm_config_userconfig: string;
		npm_config_local_prefix: string;
		SYSTEMD_EXEC_PID: string;
		DBUS_SESSION_BUS_ADDRESS: string;
		GSM_SKIP_SSH_AGENT_WORKAROUND: string;
		DBUS_STARTER_BUS_TYPE: string;
		COLORTERM: string;
		COLOR: string;
		NVM_DIR: string;
		WAYLAND_DISPLAY: string;
		LOGNAME: string;
		_: string;
		PIP_USER: string;
		npm_config_prefix: string;
		npm_config_npm_version: string;
		XDG_SESSION_CLASS: string;
		MEMORY_PRESSURE_WATCH: string;
		USERNAME: string;
		TERM: string;
		OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE: string;
		npm_config_cache: string;
		GNOME_DESKTOP_SESSION_ID: string;
		RUSTUP_HOME: string;
		npm_config_node_gyp: string;
		PATH: string;
		SESSION_MANAGER: string;
		NODE: string;
		npm_package_name: string;
		GNOME_TERMINAL_SCREEN: string;
		TRANSFORMERS_CACHE: string;
		XDG_MENU_PREFIX: string;
		GNOME_SETUP_DISPLAY: string;
		XDG_RUNTIME_DIR: string;
		COREPACK_ENABLE_AUTO_PIN: string;
		DISPLAY: string;
		LANG: string;
		XDG_CURRENT_DESKTOP: string;
		PYTHONUSERBASE: string;
		NoDefaultCurrentDirectoryInExePath: string;
		XDG_SESSION_DESKTOP: string;
		GNOME_TERMINAL_SERVICE: string;
		XMODIFIERS: string;
		XAUTHORITY: string;
		LS_COLORS: string;
		npm_lifecycle_script: string;
		SSH_AUTH_SOCK: string;
		SHELL: string;
		GOPATH: string;
		npm_package_version: string;
		npm_lifecycle_event: string;
		QT_ACCESSIBILITY: string;
		GDMSESSION: string;
		CLAUDECODE: string;
		QT_IM_MODULE: string;
		npm_config_globalconfig: string;
		npm_config_init_module: string;
		PWD: string;
		npm_execpath: string;
		XDG_DATA_DIRS: string;
		DBUS_STARTER_ADDRESS: string;
		CARGO_HOME: string;
		NVM_CD_FLAGS: string;
		npm_config_global_prefix: string;
		npm_command: string;
		CONDA_PREFIX: string;
		VTE_VERSION: string;
		MEMORY_PRESSURE_WRITE: string;
		INIT_CWD: string;
		EDITOR: string;
		NODE_ENV: string;
		[key: `PUBLIC_${string}`]: undefined;
		[key: `${string}`]: string | undefined;
	}
}

/**
 * Similar to [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), but only includes variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Note that public dynamic environment variables must all be sent from the server to the client, causing larger network requests — when possible, use `$env/static/public` instead.
 * 
 * ```ts
 * import { env } from '$env/dynamic/public';
 * console.log(env.PUBLIC_DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 */
declare module '$env/dynamic/public' {
	export const env: {
		[key: `PUBLIC_${string}`]: string | undefined;
	}
}
