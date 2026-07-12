// linter complains about mixed type exports so separated them
// this file contains the non-component exports
import { createContext, useContext } from 'react';

// holds whichever api module (httpApi or demoApi) main.jsx injects via provider
export const ApiContext = createContext(null);

// lets any component read the injected api without importing httpapi/demoapi directly
export function useApi() {
  const api = useContext(ApiContext);
  if (!api) throw new Error('useApi must be used within an ApiProvider');
  return api;
}