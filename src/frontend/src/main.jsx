import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App';
import { ApiProvider } from './api/apiProvider';
import * as httpApi from './api/httpApi';
import * as demoApi from './api/demoApi';

const api = import.meta.env.MODE === 'demo' ? demoApi : httpApi;

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ApiProvider api={api}>
      <App />
    </ApiProvider>
  </StrictMode>
);
