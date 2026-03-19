// SafeRoute AI — Service Worker v1
const CACHE = 'saferoute-v1';
const SHELL = ['/', '/index.html', '/manifest.json'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  // Always use network for API calls — predictions must be live
  if (url.pathname.startsWith('/predict') ||
      url.pathname.startsWith('/traffic') ||
      url.hostname.includes('onrender.com') ||
      url.hostname.includes('openstreetmap.org') ||
      url.hostname.includes('openweathermap.org') ||
      url.hostname.includes('project-osrm.org') ||
      url.hostname.includes('overpass-api.de')) {
    return;
  }
  // Cache-first for app shell
  e.respondWith(
    caches.match(e.request).then(cached => {
      if (cached) return cached;
      return fetch(e.request).then(response => {
        if (!response || response.status !== 200) return response;
        const clone = response.clone();
        caches.open(CACHE).then(c => c.put(e.request, clone));
        return response;
      }).catch(() => caches.match('/index.html'));
    })
  );
});