// uses plotly.js-dist-min instead of the full bundle (x3 as big) bundled by
// react-plotly.js by default
// see: https://github.com/plotly/react-plotly.js#customizing-the-plotlyjs-bundle
import Plotly from 'plotly.js-dist-min';
import _factory from 'react-plotly.js/factory';

// returns random type, fixed for now but need to research why
const factory = typeof _factory === 'function' ? _factory : _factory.default;
const Plot = factory(Plotly);
export default Plot;