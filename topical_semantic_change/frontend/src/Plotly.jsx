// Uses plotly.js-dist-min (~1MB) instead of the full bundle (~3MB) bundled by
// react-plotly.js by default. See: https://github.com/plotly/react-plotly.js#customizing-the-plotlyjs-bundle
import Plotly from 'plotly.js-dist-min';
import _factory from 'react-plotly.js/factory';

const factory = typeof _factory === 'function' ? _factory : _factory.default;
const Plot = factory(Plotly);
export default Plot;