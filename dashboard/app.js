const state = { recommendations: [], elasticity: [], meals: [], metrics: {}, guardrail: 20, demandCap: 15, sensitivity: 1 };

const $ = (selector) => document.querySelector(selector);
const money = (value) => `$${Math.round(value).toLocaleString('en-US')}`;
const percent = (value, digits = 1) => `${value >= 0 ? '+' : ''}${(value * 100).toFixed(digits)}%`;

async function loadCsv(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path}`);
  const [header, ...rows] = (await response.text()).trim().split(/\r?\n/).map((row) => row.split(','));
  return rows.map((row) => Object.fromEntries(header.map((key, index) => [key, row[index]])));
}

function numericRows(rows, fields) { return rows.map((row) => Object.fromEntries(Object.entries(row).map(([key, value]) => [key, fields.includes(key) ? Number(value) : value]))); }

async function initialize() {
  try {
    const [recommendations, elasticity, meals, metrics] = await Promise.all([
      loadCsv('../outputs/recommendations/pricing_recommendations.csv'), loadCsv('../outputs/recommendations/meal_elasticity.csv'), loadCsv('../data/raw/meal_info.csv'), loadCsv('../outputs/recommendations/model_metrics.csv')
    ]);
    state.recommendations = numericRows(recommendations, ['center_id', 'meal_id', 'current_price', 'recommended_price', 'price_change_pct', 'expected_demand', 'expected_revenue', 'demand_loss_vs_current']);
    state.elasticity = numericRows(elasticity, ['meal_id', 'observations', 'price_points', 'elasticity']);
    state.meals = meals;
    state.metrics = numericRows(metrics, ['MAE', 'RMSE', 'R2', 'train_through_week', 'test_weeks'])[0];
    renderAll();
  } catch (error) {
    console.error(error);
    $('#overview-recommendations').innerHTML = '<div class="empty-state">Run the pipeline from the project root, then refresh this page.</div>';
  }
}

function renderAll() { updateMetrics(); renderRecommendations(); renderOverviewCards(); drawChart('#demand-chart', 0); updatePreview(); lucide.createIcons(); }

function updateMetrics() {
  const rows = state.recommendations; const totalRevenue = rows.reduce((sum, row) => sum + row.expected_revenue, 0); const avgMove = rows.reduce((sum, row) => sum + row.price_change_pct, 0) / rows.length; const safe = rows.filter((row) => row.demand_loss_vs_current <= state.demandCap / 100).length;
  $('#metric-r2').textContent = state.metrics.R2.toFixed(3); $('#pipeline-r2').textContent = state.metrics.R2.toFixed(3); $('#metric-revenue').textContent = money(totalRevenue / Math.max(rows.length, 1)); $('#metric-recs').textContent = rows.length.toLocaleString(); $('#metric-price-change').textContent = percent(avgMove); $('#revenue-change').textContent = percent(avgMove * .72); $('#guardrail-count').textContent = `${Math.round((safe / Math.max(rows.length, 1)) * 100)}%`; $('#center-count').textContent = new Set(rows.map((row) => row.center_id)).size; $('#meal-count').textContent = new Set(rows.map((row) => row.meal_id)).size;
}

function mealInfo(mealId) { return state.meals.find((meal) => Number(meal.meal_id) === Number(mealId)) || {}; }
function elasticityFor(mealId) { return state.elasticity.find((item) => Number(item.meal_id) === Number(mealId)) || { elasticity: -1.8, elasticity_band: 'stable' }; }

function renderOverviewCards() {
  const rows = state.recommendations.slice(0, 3); $('#overview-recommendations').innerHTML = rows.map((row) => { const info = mealInfo(row.meal_id); return `<div class="rec-card"><div class="rec-top"><strong>Meal ${row.meal_id}</strong><span>Center ${row.center_id}</span></div><div class="rec-price"><strong>${money(row.recommended_price)}</strong><span>${percent(row.price_change_pct)}</span></div><div class="rec-foot"><span>${info.category || 'Meal'} · ${info.cuisine || 'Food'}</span><b>${money(row.expected_revenue)} revenue</b></div></div>`; }).join('');
}

function filteredRows() { const query = ($('#search-input')?.value || '').toLowerCase(); const band = $('#band-filter')?.value || 'all'; return state.recommendations.filter((row) => { const info = mealInfo(row.meal_id); const item = elasticityFor(row.meal_id); const haystack = `${row.meal_id} ${row.center_id} ${info.category || ''} ${info.cuisine || ''}`.toLowerCase(); return haystack.includes(query) && (band === 'all' || item.elasticity_band === band); }); }

function renderRecommendations() { const rows = filteredRows().slice(0, 100); $('#recommendations-body').innerHTML = rows.map((row) => { const info = mealInfo(row.meal_id); return `<tr><td class="meal-cell"><strong>Meal ${row.meal_id}</strong><small>Center ${row.center_id} · ${info.category || 'Uncategorized'}</small></td><td class="price-cell">${money(row.current_price)}</td><td class="price-cell"><b>${money(row.recommended_price)}</b></td><td><span class="move-pill">${percent(row.price_change_pct)}</span></td><td>${Math.round(row.expected_demand).toLocaleString()}</td><td><b>${money(row.expected_revenue)}</b></td><td class="loss-safe">${(row.demand_loss_vs_current * 100).toFixed(1)}%</td><td><button class="row-action" title="Open scenario"><i data-lucide="arrow-up-right"></i></button></td></tr>`; }).join(''); $('#result-count').textContent = `${filteredRows().length.toLocaleString()} recommendations`; $('#table-footer-count').textContent = `Showing ${rows.length.toLocaleString()} of ${filteredRows().length.toLocaleString()}`; lucide.createIcons(); }

function pathFor(points, width, height, pad) { return points.map((point, index) => `${index ? 'L' : 'M'} ${pad + point.x * (width - pad * 2)} ${height - pad - point.y * (height - pad * 2)}`).join(' '); }
function drawChart(selector, rowIndex = 0) { const svg = $(selector); if (!svg) return; const row = state.recommendations[rowIndex] || state.recommendations[0]; if (!row) return; const elasticity = Math.abs(elasticityFor(row.meal_id).elasticity || 1.8) * state.sensitivity; const width = 620; const height = 220; const pad = 27; const points = Array.from({ length: 18 }, (_, index) => { const x = index / 17; const move = (x - .5) * .4; const y = Math.max(.18, Math.min(1, .72 - move * elasticity * .64)); return { x, y }; }); const recommendedX = Math.min(1, Math.max(0, (row.recommended_price / row.current_price - .8) / .4)); const grid = [0, .5, 1].map((value) => `<line x1="${pad}" x2="${width - pad}" y1="${height - pad - value * (height - pad * 2)}" y2="${height - pad - value * (height - pad * 2)}" stroke="#edf0f4" stroke-dasharray="3 4"/><text x="0" y="${height - pad - value * (height - pad * 2) + 4}" fill="#a4adba" font-size="10">${Math.round(value * row.expected_demand).toLocaleString()}</text>`).join(''); svg.innerHTML = `${grid}<path d="${pathFor(points, width, height, pad)}" fill="none" stroke="#778eff" stroke-width="3" stroke-linecap="round"/><line x1="${pad + recommendedX * (width - pad * 2)}" x2="${pad + recommendedX * (width - pad * 2)}" y1="${pad}" y2="${height - pad}" stroke="#f4bd4f" stroke-dasharray="4 5"/><circle cx="${pad + recommendedX * (width - pad * 2)}" cy="${height - pad - points[Math.round(recommendedX * 17)].y * (height - pad * 2)}" r="5" fill="#f4bd4f" stroke="#fff" stroke-width="3"/><text x="${pad}" y="${height - 5}" fill="#9aa4b2" font-size="10">−20%</text><text x="${width - pad - 22}" y="${height - 5}" fill="#9aa4b2" font-size="10">+20%</text>`; $('#chart-caption').textContent = `${percent(row.price_change_pct)} price → ${(row.demand_loss_vs_current * 100).toFixed(1)}% demand loss`; }

function updatePreview() { const row = state.recommendations[0]; if (!row) return; const move = Math.min(state.guardrail / 100, row.price_change_pct); const loss = Math.abs(elasticityFor(row.meal_id).elasticity || 1.8) * state.sensitivity * move * .42; const revenue = row.current_price * (1 + move) * row.expected_demand * (1 - loss); $('#preview-move').textContent = percent(move); $('#preview-revenue').textContent = money(revenue); $('#preview-demand').textContent = `−${(loss * 100).toFixed(1)}%`; drawChart('#preview-chart', 0); $('#decision-title').textContent = loss <= state.demandCap / 100 ? 'Revenue wins inside the guardrail' : 'Demand cap is blocking this move'; $('#decision-copy').textContent = loss <= state.demandCap / 100 ? 'The current model finds room to increase price while protecting modeled demand.' : 'Reduce sensitivity or widen the demand cap to inspect a less conservative scenario.'; }

function showView(view) { document.querySelectorAll('.view').forEach((item) => item.classList.toggle('is-visible', item.id === `${view}-view`)); document.querySelectorAll('.nav-item').forEach((item) => item.classList.toggle('is-active', item.dataset.view === view)); $('#page-title').textContent = view === 'studio' ? 'Model studio' : view === 'recommendations' ? 'Recommendations' : 'Overview'; }
document.querySelectorAll('[data-view]').forEach((button) => button.addEventListener('click', () => showView(button.dataset.view))); document.querySelectorAll('[data-view-target]').forEach((button) => button.addEventListener('click', () => showView(button.dataset.viewTarget)));
const presets = { recommended: { sensitivity: 1, guardrail: 20, demandCap: 15 }, careful: { sensitivity: .8, guardrail: 10, demandCap: 10 }, growth: { sensitivity: 1.2, guardrail: 25, demandCap: 20 } };
document.querySelectorAll('[data-preset]').forEach((button) => button.addEventListener('click', () => { const preset = presets[button.dataset.preset]; state.sensitivity = preset.sensitivity; state.guardrail = preset.guardrail; state.demandCap = preset.demandCap; $('#elasticity').value = preset.sensitivity; $('#price-guardrail').value = preset.guardrail; $('#demand-cap').value = preset.demandCap; $('#elasticity-output').textContent = `${preset.sensitivity.toFixed(2)}x`; $('#price-guardrail-output').textContent = `${preset.guardrail}%`; $('#demand-cap-output').textContent = `${preset.demandCap}%`; document.querySelectorAll('[data-preset]').forEach((item) => item.classList.toggle('is-selected', item === button)); updatePreview(); updateMetrics(); }));
['search-input', 'band-filter'].forEach((id) => document.getElementById(id)?.addEventListener('input', renderRecommendations));
[['elasticity', 'elasticity-output', (value) => `${Number(value).toFixed(2)}x`], ['price-guardrail', 'price-guardrail-output', (value) => `${value}%`], ['demand-cap', 'demand-cap-output', (value) => `${value}%`]].forEach(([id, output, formatter]) => document.getElementById(id).addEventListener('input', (event) => { const value = Number(event.target.value); document.getElementById(output).textContent = formatter(value); if (id === 'elasticity') state.sensitivity = value; if (id === 'price-guardrail') state.guardrail = value; if (id === 'demand-cap') state.demandCap = value; updatePreview(); updateMetrics(); }));
document.querySelectorAll('.toggle').forEach((toggle) => toggle.addEventListener('click', () => { toggle.classList.toggle('is-on'); toggle.setAttribute('aria-pressed', toggle.classList.contains('is-on')); })); $('#run-scenario').addEventListener('click', () => { updatePreview(); $('#save-note').innerHTML = '<i data-lucide="check"></i> Scenario ready for review. Human approval is still required.'; lucide.createIcons(); $('#toast').classList.add('is-visible'); setTimeout(() => $('#toast').classList.remove('is-visible'), 2400); }); $('#export-csv').addEventListener('click', () => { const rows = filteredRows(); const header = 'meal_id,center_id,current_price,recommended_price,price_change_pct,expected_demand,expected_revenue,demand_loss_vs_current'; const csv = [header, ...rows.map((row) => [row.meal_id, row.center_id, row.current_price, row.recommended_price, row.price_change_pct, row.expected_demand, row.expected_revenue, row.demand_loss_vs_current].join(','))].join('\n'); const link = document.createElement('a'); link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' })); link.download = 'pulse-recommendations.csv'; link.click(); URL.revokeObjectURL(link.href); });
initialize();