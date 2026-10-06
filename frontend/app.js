/**
 * Predictive Maintenance Diagnostic System
 * Frontend Client Controller
 * Course: IT3051 - Fundamentals of Data Mining (SLIIT)
 * Group: 05 - 'Cognita'
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const backendStatusDot = document.getElementById('backendStatusDot');
  const backendStatusText = document.getElementById('backendStatusText');
  const telemetryForm = document.getElementById('telemetryForm');
  const btnDiagnose = document.getElementById('btnDiagnose');
  const btnDiagnoseText = document.getElementById('btnDiagnoseText');
  const diagnoseSpinner = document.getElementById('diagnoseSpinner');
  const btnReset = document.getElementById('btnReset');
  const validationAlert = document.getElementById('validationAlert');
  const validationMessage = document.getElementById('validationMessage');

  // Input Fields
  const machineType = document.getElementById('machineType');
  const controlMode = document.getElementById('controlMode');
  const airTemp = document.getElementById('airTemp');
  const processTemp = document.getElementById('processTemp');
  const rotSpeed = document.getElementById('rotSpeed');
  const torque = document.getElementById('torque');
  const toolWear = document.getElementById('toolWear');

  // Results Containers
  const resultsStatusPill = document.getElementById('resultsStatusPill');
  const emptyResultsState = document.getElementById('emptyResultsState');
  const predictionDisplay = document.getElementById('predictionDisplay');
  const outcomeBanner = document.getElementById('outcomeBanner');
  const outcomeIcon = document.getElementById('outcomeIcon');
  const predictedClass = document.getElementById('predictedClass');
  const predictedSeverity = document.getElementById('predictedSeverity');
  const confidenceVal = document.getElementById('confidenceVal');
  const diagnosisText = document.getElementById('diagnosisText');
  const recommendationText = document.getElementById('recommendationText');
  const valDeltaT = document.getElementById('valDeltaT');
  const valMechPower = document.getElementById('valMechPower');
  const valOverstrain = document.getElementById('valOverstrain');
  const valMissingCount = document.getElementById('valMissingCount');
  const probabilityBars = document.getElementById('probabilityBars');
  const inputEchoTags = document.getElementById('inputEchoTags');

  let activePresetButtons = document.querySelectorAll('.preset-btn');
  let schemaPresets = null;

  // 1. Check System Health & Load Schema
  async function initBackendConnection() {
    try {
      const healthRes = await fetch('/health');
      if (healthRes.ok) {
        const healthData = await healthRes.json();
        backendStatusDot.className = 'status-indicator-dot online';
        backendStatusText.textContent = 'Operational • Ready';
      } else {
        throw new Error('Health check returned non-200 status');
      }

      // Fetch Schema & Presets
      const schemaRes = await fetch('/api/v1/schema');
      if (schemaRes.ok) {
        const schemaData = await schemaRes.json();
        schemaPresets = schemaData.preset_scenarios;
      }
    } catch (err) {
      console.warn('Backend connection warning:', err);
      backendStatusDot.className = 'status-indicator-dot offline';
      backendStatusText.textContent = 'Offline (Check Server)';
    }
  }

  // 2. Scenario Presets Handler
  activePresetButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const presetKey = btn.getAttribute('data-preset');
      hideValidationError();

      activePresetButtons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');

      if (schemaPresets && schemaPresets[presetKey]) {
        applyPreset(schemaPresets[presetKey].payload);
      } else {
        // Fallback local presets
        applyFallbackPreset(presetKey);
      }
    });
  });

  function applyPreset(payload) {
    machineType.value = payload.type || 'L';
    controlMode.value = payload.control || 'A';
    airTemp.value = payload.air_temperature_k !== null && payload.air_temperature_k !== undefined ? payload.air_temperature_k : '';
    processTemp.value = payload.process_temperature_k !== null && payload.process_temperature_k !== undefined ? payload.process_temperature_k : '';
    rotSpeed.value = payload.rotational_speed_rpm !== null && payload.rotational_speed_rpm !== undefined ? payload.rotational_speed_rpm : '';
    torque.value = payload.torque_nm !== null && payload.torque_nm !== undefined ? payload.torque_nm : '';
    toolWear.value = payload.tool_wear_min !== null && payload.tool_wear_min !== undefined ? payload.tool_wear_min : '';
  }

  function applyFallbackPreset(key) {
    const fallbacks = {
      normal: { type: 'L', control: 'A', air_temperature_k: 300.0, process_temperature_k: 310.5, rotational_speed_rpm: 1520.0, torque_nm: 38.5, tool_wear_min: 45.0 },
      power_failure: { type: 'L', control: 'B', air_temperature_k: null, process_temperature_k: null, rotational_speed_rpm: 1350.0, torque_nm: 68.2, tool_wear_min: null },
      heat_dissipation: { type: 'M', control: 'A', air_temperature_k: 303.2, process_temperature_k: 309.8, rotational_speed_rpm: 1320.0, torque_nm: 48.0, tool_wear_min: 70.0 },
      overstrain: { type: 'L', control: 'C', air_temperature_k: null, process_temperature_k: null, rotational_speed_rpm: null, torque_nm: 60.5, tool_wear_min: 215.0 },
      tool_wear: { type: 'H', control: 'C', air_temperature_k: null, process_temperature_k: null, rotational_speed_rpm: null, torque_nm: 42.0, tool_wear_min: 235.0 },
      control_b_missing: { type: 'M', control: 'B', air_temperature_k: null, process_temperature_k: null, rotational_speed_rpm: 1600.0, torque_nm: 35.0, tool_wear_min: null },
    };
    if (fallbacks[key]) applyPreset(fallbacks[key]);
  }

  // 3. Client-side Validation
  function validateInputs() {
    hideValidationError();

    if (!machineType.value) {
      showValidationError('Please select a valid Machine Variant Type.');
      machineType.focus();
      return false;
    }

    if (!controlMode.value) {
      showValidationError('Please select a Diagnostic Control Mode.');
      controlMode.focus();
      return false;
    }

    const aVal = airTemp.value !== '' ? parseFloat(airTemp.value) : null;
    const pVal = processTemp.value !== '' ? parseFloat(processTemp.value) : null;
    const sVal = rotSpeed.value !== '' ? parseFloat(rotSpeed.value) : null;
    const tVal = torque.value !== '' ? parseFloat(torque.value) : null;
    const wVal = toolWear.value !== '' ? parseFloat(toolWear.value) : null;

    if (aVal !== null && (isNaN(aVal) || aVal < 280 || aVal > 340)) {
      showValidationError('Air Temperature must be a valid number between 280.0 K and 340.0 K.');
      airTemp.focus();
      return false;
    }

    if (pVal !== null && (isNaN(pVal) || pVal < 280 || pVal > 350)) {
      showValidationError('Process Temperature must be a valid number between 280.0 K and 350.0 K.');
      processTemp.focus();
      return false;
    }

    if (aVal !== null && pVal !== null && pVal < (aVal - 5.0)) {
      showValidationError(`Thermodynamic inconsistency: Process Temperature (${pVal} K) cannot be significantly lower than Air Temperature (${aVal} K).`);
      processTemp.focus();
      return false;
    }

    if (sVal !== null && (isNaN(sVal) || sVal < 800 || sVal > 4000)) {
      showValidationError('Rotational Speed must be a valid number between 800 and 4000 RPM.');
      rotSpeed.focus();
      return false;
    }

    if (tVal !== null && (isNaN(tVal) || tVal < 0 || tVal > 150)) {
      showValidationError('Spindle Torque must be a valid number between 0.0 and 150.0 Nm.');
      torque.focus();
      return false;
    }

    if (wVal !== null && (isNaN(wVal) || wVal < 0 || wVal > 350)) {
      showValidationError('Tool Wear must be a valid number between 0 and 350 minutes.');
      toolWear.focus();
      return false;
    }

    return true;
  }

  function showValidationError(msg) {
    validationMessage.textContent = msg;
    validationAlert.classList.remove('hidden');
  }

  function hideValidationError() {
    validationAlert.classList.add('hidden');
    validationMessage.textContent = '';
  }

  // 4. Form Submission & Inference API Call
  telemetryForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (!validateInputs()) {
      return;
    }

    // Set Loading State
    btnDiagnose.disabled = true;
    diagnoseSpinner.classList.remove('hidden');
    btnDiagnoseText.textContent = 'Processing Inference...';
    resultsStatusPill.textContent = 'Analyzing...';

    // Construct Payload
    const payload = {
      type: machineType.value,
      control: controlMode.value,
      air_temperature_k: airTemp.value !== '' ? parseFloat(airTemp.value) : null,
      process_temperature_k: processTemp.value !== '' ? parseFloat(processTemp.value) : null,
      rotational_speed_rpm: rotSpeed.value !== '' ? parseFloat(rotSpeed.value) : null,
      torque_nm: torque.value !== '' ? parseFloat(torque.value) : null,
      tool_wear_min: toolWear.value !== '' ? parseFloat(toolWear.value) : null,
    };

    try {
      const response = await fetch('/api/v1/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        let errMsg = errorData.detail || 'Inference failed on the backend server.';
        if (Array.isArray(errorData.detail)) {
          errMsg = errorData.detail.map((d) => d.msg).join('; ');
        }
        throw new Error(errMsg);
      }

      const result = await response.json();
      renderPredictionResult(result);
    } catch (err) {
      console.error('Prediction API Error:', err);
      showValidationError(`Server Error: ${err.message}`);
      resultsStatusPill.textContent = 'Error Encountered';
    } finally {
      btnDiagnose.disabled = false;
      diagnoseSpinner.classList.add('hidden');
      btnDiagnoseText.innerHTML = '&#9654; Run Diagnostic Analysis';
    }
  });

  // 5. Render Diagnostic Results
  function renderPredictionResult(res) {
    emptyResultsState.classList.add('hidden');
    predictionDisplay.classList.remove('hidden');

    const predClass = res.predicted_class;
    const severity = res.severity || (res.is_failure ? 'CRITICAL' : 'NORMAL');

    // Update Outcome Banner
    predictedClass.textContent = predClass;
    confidenceVal.textContent = `${res.confidence.toFixed(1)}%`;
    predictedSeverity.textContent = `Severity Level: ${severity}`;
    resultsStatusPill.textContent = res.is_failure ? 'Failure Detected' : 'Normal Operation';

    outcomeBanner.className = 'outcome-banner';
    if (severity === 'NORMAL') {
      outcomeBanner.classList.add('normal');
      outcomeIcon.innerHTML = '&#10004;';
    } else if (severity === 'CRITICAL') {
      outcomeBanner.classList.add('critical');
      outcomeIcon.innerHTML = '&#9888;';
    } else {
      outcomeBanner.classList.add('warning');
      outcomeIcon.innerHTML = '&#9888;';
    }

    // Diagnosis & Recommendation
    diagnosisText.textContent = res.diagnosis;
    recommendationText.textContent = res.recommended_action;

    // Derived Physics
    const physics = res.computed_physics;
    valDeltaT.textContent = physics.temp_difference_k !== null ? `${physics.temp_difference_k.toFixed(1)} K` : 'Unmeasured';
    valMechPower.textContent = physics.mechanical_power_w !== null ? `${physics.mechanical_power_w.toLocaleString()} W` : 'Unmeasured';
    valOverstrain.textContent = physics.overstrain_product !== null ? `${physics.overstrain_product.toLocaleString()} min·Nm` : 'Unmeasured';
    valMissingCount.textContent = `${physics.missing_sensors_count} / 5 unmeasured`;

    // Probability Bars
    probabilityBars.innerHTML = '';
    const sortedClasses = Object.entries(res.class_probabilities).sort((a, b) => b[1] - a[1]);

    sortedClasses.forEach(([clsName, prob]) => {
      const isWinner = clsName === predClass;
      const row = document.createElement('div');
      row.className = `prob-row ${isWinner ? 'active' : ''}`;
      if (isWinner) {
        if (severity === 'CRITICAL') row.classList.add('critical');
        else if (severity === 'NORMAL') row.classList.add('normal');
      }

      row.innerHTML = `
        <div class="prob-header">
          <span class="prob-name">${clsName}</span>
          <span class="prob-pct">${prob.toFixed(2)}%</span>
        </div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: ${Math.max(prob, 0.5)}%"></div>
        </div>
      `;
      probabilityBars.appendChild(row);
    });

    // Input Echo Summary
    inputEchoTags.innerHTML = '';
    const summary = res.input_summary || {};
    const labels = {
      Type: 'Type',
      Control: 'Control',
      'Air temperature (K)': 'Air Temp',
      'Process temperature (K)': 'Process Temp',
      'Rotational speed (rpm)': 'Speed',
      'Torque (Nm)': 'Torque',
      'Tool wear (min)': 'Wear',
    };

    for (const [k, v] of Object.entries(summary)) {
      const tag = document.createElement('span');
      tag.className = 'echo-tag';
      const label = labels[k] || k;
      const valStr = v !== null && v !== undefined ? v : 'NaN';
      tag.innerHTML = `${label}: <strong>${valStr}</strong>`;
      inputEchoTags.appendChild(tag);
    }
  }

  // 6. Reset Handler
  btnReset.addEventListener('click', () => {
    telemetryForm.reset();
    activePresetButtons.forEach((b) => b.classList.remove('active'));
    hideValidationError();
    predictionDisplay.classList.add('hidden');
    emptyResultsState.classList.remove('hidden');
    resultsStatusPill.textContent = 'Awaiting Telemetry';
  });

  // Initialize
  initBackendConnection();
});
