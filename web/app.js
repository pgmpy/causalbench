const DGM_META = {
    "linear_gaussian": {
        name: "Linear Gaussian",
        equation: '\\( X = \\alpha^T Z + \\varepsilon_1;\\; Y = \\beta^T Z + \\text{effect\_size}\\cdot X + \\varepsilon_2 \\)'
    },
    "nonlinear_gaussian": {
        name: "Nonlinear Gaussian",
        equation: '\\( X = \\sin(\\text{effect\_size} \\cdot \\sum_j Z_j) + \\varepsilon_1;\\; Y = \\exp(\\text{effect\_size} \\cdot \\sum_j Z_j \\cdot 0.2) + \\varepsilon_2 \\)'
    },
    "discrete_categorical": {
        name: "Discrete Categorical",
        equation: '\\( Z_j \\sim \\mathrm{DiscreteUniform}(0, n_{\\text{categories}}-1);\\; X = \\sum_j Z_j + \\text{noise};\\; Y = \\sum_j Z_j + \\text{noise} \\)'
    },
    "mixed_data": {
        name: "Mixed Data",
        equation: '\\( X = Z^T \\alpha + \\varepsilon_1;\\; Y = Z^T \\beta + \\text{effect\_size} \\cdot X + \\varepsilon_2 \\)'
    },
    "non_gaussian_continuous": {
        name: "Non-Gaussian Continuous",
        equation: '\\( X = |Z^T \\alpha| + e_1,\\; e_1 \\sim \\mathrm{Exponential}(1.0);\\; Y = (Z^T \\beta)^2 + \\text{effect\\_size} \\cdot X + e_2,\\; e_2 \\sim \\mathrm{Exponential}(1.0) \\)'
    }
};

let allData = [];
let dgmMap = {};
let chartCalibration, chartPower;

const dgmSelect = document.getElementById('dgm-select');
const sampleSizeSelect = document.getElementById('sample-size-select');
const csvInput = document.getElementById('csv-input');
const dgmEquationEl = document.getElementById('dgm-equation');

const DEFAULT_CSV = "results/ci_benchmark_summaries.csv";

window.addEventListener('DOMContentLoaded', () => {
    fetch(DEFAULT_CSV)
        .then(res => {
            if (!res.ok) throw new Error("Default CSV not found");
            return res.text();
        })
        .then(csvText => {
            allData = parseCSV(csvText);
            buildDGMMap(allData);
            populateDropdowns();
            renderCharts();
        })
        .catch(() => {
            // file not found, don't crash
        });
});

csvInput.addEventListener('change', handleCSVUpload);
dgmSelect.addEventListener('change', onDGMChange);
sampleSizeSelect.addEventListener('change', renderCharts);

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
        const { dgm, sample_size } = row;
        if (!dgmMap[dgm]) dgmMap[dgm] = new Set();
        dgmMap[dgm].add(row.sample_size);
    });
    Object.keys(dgmMap).forEach(dgm => {
        dgmMap[dgm] = Array.from(dgmMap[dgm]).sort((a, b) => a - b);
    });
}

function setSelectOptions(select, list) {
    select.innerHTML = '';
    for (const obj of list) {
        const option = document.createElement('option');
        option.value = obj.value;
        option.textContent = obj.label;
        select.appendChild(option);
    }
}

function populateDropdowns() {
    setSelectOptions(dgmSelect, Object.keys(dgmMap).map(d => ({ value: d, label: DGM_META[d]?.name || d })));
    onDGMChange();
}

function onDGMChange() {
    const dgm = dgmSelect.value;
    const sizes = dgm ? dgmMap[dgm] : [];
    setSelectOptions(sampleSizeSelect, sizes.map(s => ({ value: s, label: s })));
    updateDGMEquation();
    renderCharts();
}

function updateDGMEquation() {
    const dgm = dgmSelect.value;
    dgmEquationEl.innerHTML = DGM_META[dgm]?.equation || '';
    if (window.MathJax) MathJax.typesetPromise([dgmEquationEl]);
}

function renderCharts() {
    if (!allData.length) return;
    const dgm = dgmSelect.value;
    const sampleSize = Number(sampleSizeSelect.value);

    // Calibration plot: Type I error vs significance_level, effect_size == 0
    const calibData = allData.filter(r => r.dgm === dgm && r.sample_size === sampleSize && r.effect_size === 0);
    const calibSLs = calibData.map(r => r.significance_level);
    const calibT1 = calibData.map(r => r.type1_error);

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
            plugins: { legend: { display: true } },
            scales: {
                x: { title: { display: true, text: 'Significance Level' }, min: 0, max: 1 },
                y: { title: { display: true, text: 'Type I Error' }, min: 0, max: 1 }
            }
        }
    });

    // Power plot: Power vs effect size, significance_level == 0.05
    const powerData = allData.filter(row =>
        row.dgm === dgm &&
        row.sample_size === sampleSize &&
        row.significance_level === 0.05
    );
    // Group by ci_test (for legend, since benchmark may run several tests)
    const ciTestsPower = Array.from(new Set(powerData.map(r => r.ci_test)));
    const powerDatasets = ciTestsPower.map((ciTest, idx) => {
        const rows = powerData.filter(r => r.ci_test === ciTest).sort((a, b) => a.effect_size - b.effect_size);
        return {
            label: ciTest,
            data: rows.map(r => ({ x: r.effect_size, y: r.power })),
            borderColor: chartColor(idx, 1),
            backgroundColor: chartColor(idx, 0.2),
            pointRadius: 3,
            fill: false
        };
    });

    if (chartPower) chartPower.destroy();
    chartPower = new Chart(document.getElementById('power-plot').getContext('2d'), {
        type: 'line',
        data: {
            datasets: powerDatasets
        },
        options: {
            responsive: true,
            plugins: { legend: { display: true } },
            scales: {
                x: { title: { display: true, text: 'Effect Size' }, min: 0, max: 1 },
                y: { title: { display: true, text: 'Power' }, min: 0, max: 1 }
            }
        }
    });
}

function chartColor(idx, alpha = 1) {
    const colors = [
        `rgba(54, 162, 235, ${alpha})`,   // blue
        `rgba(255,99,132,${alpha})`,      // red
        `rgba(255, 205, 86, ${alpha})`,   // yellow
        `rgba(75, 192, 192, ${alpha})`,   // teal
        `rgba(153, 102, 255, ${alpha})`,  // purple
        `rgba(201, 203, 207, ${alpha})`,  // grey
        `rgba(0, 200, 83, ${alpha})`,     // green
        `rgba(255, 87, 34, ${alpha})`,    // deep orange
    ];
    return colors[idx % colors.length];
}
