// REVIEW react-plotly.js is very large by default
// meant to force plotly.js-dist-min instead, as described here
// https://community.plotly.com/t/how-can-i-reduce-bundle-size-of-plotly-js-in-react-app/89910/2
// remove if things aren't working???

import Plotly from 'plotly.js-dist-min';
import * as factoryNS from 'react-plotly.js/factory';

// factory.js uses __esModule exports; .default is the actual function
const Plot = factoryNS.default(Plotly);
export default Plot;