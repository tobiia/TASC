import { ApiContext } from './apiContext';

// this file contains the component exports clearly

// this file wraps <App>  to inject either httpApi or demoApi as "api"
export function ApiProvider({ api, children }) {
    return <ApiContext.Provider value={api}>{children}</ApiContext.Provider>;
}