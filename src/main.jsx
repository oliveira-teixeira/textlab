import React from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import TextAnalysisApp from './textual-analysis-app.jsx';

const root = createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <TextAnalysisApp />
  </React.StrictMode>
);
