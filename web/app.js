let allData = [];
let dgmMap = {};
let chartCalibration, chartPower;

const dgmSelect = document.getElementById('dgm-select');
const ciTestSelect = document.getElementById('ci-test-select');
const effectSizeSelect = document.getElementById('effect-size-select');
const significanceSelect = document.getElementById('significance-level-select');
const csvInput = document.getElementById('csv-input');

csvInput.addEventListener('change', handleCSVUpload);

dgmSelect.addEventListener('change', refreshControls);
ciTestSelect.addEventListener('change', refreshControls);
effectSizeSelect.addEventListener('change', refreshControls);
significanceSelect.addEventListener('change', refreshControls);

function handleCSVUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
        allData = parseCSV(e.target.result);
        buildDGMMap(allData);
        populateDropdowns();
        renderCharts();
    };
    reader.readAsText(file);
}

function parseCSV(csvText) {
    const [header, ...rows] = csvText.trim().split('\n');
    const keys = header.split(',');
    return rows.map(row =>
        Object.fromEntries(
            row.split(',').map((val, i) => [keys[i], parseValue(val)])
        )
    );
}
function parseValue(v) {
    if (!isNaN(parseFloat(v)) && isFinite(v)) return parseFloat(v);
    if (v === 'True') return true;
    if (v === 'False') return false;
    return v;
}

function buildDGMMap(data) {
    dgmMap = {};
    data.forEach(row => {
        const { dgm, ci_test, effect_size, significance_level } = row;
        if (!dgmMap[dgm]) dgmMap[dgm] = {};
        if (!dgmMap[dgm][ci_test]) dgmMap[dgm][ci_test] = {};
        if (!dgmMap[dgm][ci_test][effect_size]) dgmMap[dgm][ci_test][effect_size] = {};
        dgmMap[dgm][ci_test][effect_size][significance_level] = true;
    });
}

function populateDropdowns() {
    setSelectOptions(dgmSelect, Object.keys(dgmMap));
    onDGMChange();
}

function onDGMChange() {
    const dgm = dgmSelect.value;
    const tests = dgm ? Object.keys(dgmMap[dgm]) : [];
    setSelectOptions(ciTestSelect, tests);
    onCITestChange();
}
function onCITestChange() {
    const dgm = dgmSelect.value, test = ciTestSelect.value;
    const effects = dgm && test ? Object.keys(dgmMap[dgm][test]) : [];
    setSelectOptions(effectSizeSelect, effects);
    onEffectSizeChange();
}
function onEffectSizeChange() {
    const dgm = dgmSelect.value, test = ciTestSelect.value, effect = effectSizeSelect.value;
    const sigs = dgm && test && effect ? Object.keys(dgmMap[dgm][test][effect]) : [];
    setSelectOptions(significanceSelect, sigs);
}

dgmSelect.addEventListener('change', onDGMChange);
ciTestSelect.addEventListener('change', onCITestChange);
effectSizeSelect.addEventListener('change', onEffectSizeChange);

function setSelectOptions(select, list) {
    select.innerHTML = '';
    for (const v of list) {
        const option = document.createElement('option');
        option.value = v;
        option.textContent = v;
        select.appendChild(option);
    }
}

function refreshControls() {
    renderCharts();
}

function renderCharts() {
    if (!allData.length) return;
    const dgm = dgmSelect.value, test = ciTestSelect.value;
    const effect = parseFloat(effectSizeSelect.value);
    const significance = parseFloat(significanceSelect.value);

    const filtered = allData.filter(row =>
        row.dgm === dgm && row.ci_test === test
    );

    // Calibration Plot (Type I error vs Significance Level, effect_size == 0)
    const calibData = filtered.filter(r => r.effect_size === 0);
    const calibSLs = calibData.map(r => r.significance_level);
    const calibT1 = calibData.map(r => r.type1_error);

    // Power Plot (Power vs Sample Size, user-selected effect size & significance level)
    const powerData = allData.filter(row =>
        row.dgm === dgm &&
        row.ci_test === test &&
        row.effect_size === effect &&
        row.significance_level === significance
    );
    const sampleSizes = powerData.map(r => r.sample_size);
    const powers = powerData.map(r => r.power);

    // Rendering Calibration Plot 
    if (chartCalibration) chartCalibration.destroy();
    chartCalibration = new Chart(document.getElementById('calibration-plot').getContext('2d'), {
        type: 'line',
        data: {
            labels: calibSLs,
            datasets: [{
                label: 'Type I Error',
                data: calibT1,
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                pointRadius: 4,
                fill: true,
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { title: { display: true, text: 'Significance Level' } },
                y: { title: { display: true, text: 'Type I Error' }, min: 0, max: 1 }
            }
        }
    });

    //  Rendering Power Plot 
    if (chartPower) chartPower.destroy();
    chartPower = new Chart(document.getElementById('power-plot').getContext('2d'), {
        type: 'line',
        data: {
            labels: sampleSizes,
            datasets: [{
                label: 'Power',
                data: powers,
                borderColor: 'rgba(255,99,132,1)',
                backgroundColor: 'rgba(255,99,132,0.2)',
                pointRadius: 4,
                fill: true,
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { title: { display: true, text: 'Sample Size' } },
                y: { title: { display: true, text: 'Power' }, min: 0, max: 1 }
            }
        }
    });
}