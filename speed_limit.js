/**
 * speed_limit.js — SafeRoute AI
 * ════════════════════════════════
 * Fetches road speed limits from OpenStreetMap Overpass API
 * and triggers overspeed alerts when GPS speed exceeds the limit.
 *
 * Add to index.html:
 *   <script src="speed_limit.js"></script>
 * Then call checkSpeedLimit(lat, lng, currentSpeed) every GPS update.
 */

const SpeedLimit = (() => {
  let _lastFetch    = 0;
  let _cachedLimit  = null;
  let _alertShown   = false;
  let _alertTO      = null;

  // Default speed limits (km/h) by road type when OSM has no data
  const DEFAULTS = {
    'Highway':     100,
    'City Road':    60,
    'Residential':  30,
    'Service Road': 20,
    'Local Road':   40,
    'Unknown':      50,
  };

  /**
   * Fetch speed limit from OSM Overpass for given coordinates.
   * Throttled to once every 30 seconds.
   */
  async function fetchLimit(lat, lng) {
    const now = Date.now();
    if (_cachedLimit !== null && now - _lastFetch < 30000) {
      return _cachedLimit;
    }
    try {
      const query = `[out:json];way(around:40,${lat},${lng})[highway][maxspeed];out 1;`;
      const res   = await fetch('https://overpass-api.de/api/interpreter',
        { method:'POST', body:`data=${encodeURIComponent(query)}`,
          signal: AbortSignal.timeout(4000) });
      const data  = await res.json();
      const elem  = data?.elements?.[0];
      if (elem) {
        const raw = elem.tags?.maxspeed || '';
        // Parse "50", "50 mph", "50 km/h", "IN:urban", etc.
        if (raw.toLowerCase().includes('mph')) {
          _cachedLimit = Math.round(parseFloat(raw) * 1.60934);
        } else if (/^\d+/.test(raw)) {
          _cachedLimit = parseInt(raw);
        } else if (raw === 'IN:urban')   _cachedLimit = 50;
        else if (raw === 'IN:rural')     _cachedLimit = 100;
        else if (raw === 'IN:motorway')  _cachedLimit = 120;
        else _cachedLimit = null;
      } else {
        _cachedLimit = null;
      }
    } catch (e) {
      _cachedLimit = null;
    }
    _lastFetch = Date.now();
    return _cachedLimit;
  }

  /**
   * Main function — call every GPS update
   * @param {number} lat - current latitude
   * @param {number} lng - current longitude
   * @param {number} speedKmh - current GPS speed in km/h
   * @param {string} roadType - from fetchRoadType() e.g. "City Road"
   * @param {boolean} voiceOn - whether voice alerts are enabled
   */
  async function check(lat, lng, speedKmh, roadType, voiceOn) {
    if (!speedKmh || speedKmh < 5) return; // not moving

    // Get speed limit
    let limit = await fetchLimit(lat, lng);
    if (!limit) limit = DEFAULTS[roadType] || 50; // fallback by road type

    // Update HUD display
    updateLimitDisplay(limit, speedKmh);

    // Check overspeed
    const isOver   = speedKmh > limit;
    const isSevere = speedKmh > limit * 1.2; // 20% over = severe

    if (isOver && !_alertShown) {
      showSpeedAlert(speedKmh, limit, isSevere, voiceOn);
      _alertShown = true;
      clearTimeout(_alertTO);
      _alertTO = setTimeout(() => { _alertShown = false; }, 15000);
    } else if (!isOver && _alertShown) {
      hideSpeedAlert();
      _alertShown = false;
    }

    return { limit, isOver, isSevere };
  }

  function updateLimitDisplay(limit, speed) {
    // Update speed HUD border color based on limit
    const hud = document.getElementById('spd-hud');
    if (!hud) return;
    const ratio = speed / limit;
    if (ratio >= 1.2) {
      hud.style.borderColor = '#EA4335';
      hud.style.boxShadow   = '0 0 12px rgba(234,67,53,.4)';
    } else if (ratio >= 1.0) {
      hud.style.borderColor = '#FBBC04';
      hud.style.boxShadow   = '0 0 8px rgba(251,188,4,.3)';
    } else {
      hud.style.borderColor = 'rgba(255,255,255,0.09)';
      hud.style.boxShadow   = '0 4px 24px rgba(0,0,0,.55)';
    }

    // Update speed limit badge
    let badge = document.getElementById('speed-limit-badge');
    if (!badge) {
      badge = document.createElement('div');
      badge.id = 'speed-limit-badge';
      badge.style.cssText = `
        position:fixed;left:14px;bottom:360px;z-index:501;
        background:#fff;border:3px solid #000;border-radius:50%;
        width:46px;height:46px;display:flex;flex-direction:column;
        align-items:center;justify-content:center;
        font-family:'Google Sans',sans-serif;box-shadow:0 2px 8px rgba(0,0,0,.4);
      `;
      document.body.appendChild(badge);
    }
    badge.innerHTML = `
      <div style="font-size:8px;color:#c00;font-weight:700;line-height:1">MAX</div>
      <div style="font-size:14px;font-weight:700;color:#000;line-height:1">${limit}</div>
    `;
  }

  function showSpeedAlert(speed, limit, severe, voiceOn) {
    let alert = document.getElementById('speed-alert');
    if (!alert) {
      alert = document.createElement('div');
      alert.id = 'speed-alert';
      alert.style.cssText = `
        position:fixed;bottom:220px;left:50%;transform:translateX(-50%);
        z-index:600;background:${severe?'#EA4335':'#FBBC04'};
        color:${severe?'#fff':'#000'};border-radius:12px;
        padding:10px 20px;font-family:'Google Sans',sans-serif;
        font-size:14px;font-weight:700;box-shadow:0 4px 20px rgba(0,0,0,.4);
        display:flex;align-items:center;gap:10px;white-space:nowrap;
        transition:opacity .3s;
      `;
      document.body.appendChild(alert);
    }
    alert.innerHTML = `
      <span style="font-size:20px">${severe?'🚨':'⚠️'}</span>
      <span>${severe?'Overspeed!':'Speed limit'} ${speed} / ${limit} km/h</span>
    `;
    alert.style.opacity = '1';
    alert.style.pointerEvents = 'all';

    if (voiceOn) {
      const msg = severe
        ? `Danger! You are driving ${speed} kilometers per hour. Speed limit is ${limit}.`
        : `Speed limit is ${limit} kilometers per hour. Please slow down.`;
      const u = new SpeechSynthesisUtterance(msg);
      u.rate = 1; u.volume = 1;
      speechSynthesis.cancel();
      speechSynthesis.speak(u);
    }
  }

  function hideSpeedAlert() {
    const alert = document.getElementById('speed-alert');
    if (alert) { alert.style.opacity = '0'; alert.style.pointerEvents = 'none'; }
  }

  return { check, fetchLimit, DEFAULTS };
})();