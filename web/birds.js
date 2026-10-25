// Pájaros cúbicos de cada especie, en SVG. Los usan el juego (index.html) y el curso (aprender.html).
(function (root) {
  'use strict';
  // Cada especie es un pájaro cúbico: cuerpo, panza, ojos y un pico que la distingue.
  const BIRD = [
    { body: '#F28DB2', belly: '#FBD3E2', beak: 'flamingo', feet: '#D9578A' },
    { body: '#8A6240', belly: '#D8B98E', beak: 'small', beakc: '#F2A33A', tufts: true, big: true, feet: '#F2A33A' },
    { body: '#2C2C36', belly: '#F4F0E0', beak: 'toucan', feet: '#5A6CB0' },
    { body: '#F5C93B', belly: '#FCE79A', beak: 'duck', beakc: '#F07F1D', feet: '#F07F1D' },
    { body: '#3BAA5C', belly: '#A7DE7E', beak: 'hook', beakc: '#F2C23A', crest: '#E8463A', feet: '#555' },
    { body: '#2A3B66', belly: '#FFFFFF', beak: 'small', beakc: '#1B1B1B', wings: '#FFFFFF', feet: '#1B1B1B' },
    { body: '#D9B77E', belly: '#FFF6E2', beak: 'thin', beakc: '#6B4E2E', feet: '#A5834F' },
    { body: '#5F6B7A', belly: '#F0703A', beak: 'small', beakc: '#2A2A2A', feet: '#2A2A2A' },
  ];
  function birdSymbol(s) {
    const b = BIRD[s];
    let x = '';
    if (b.tufts) x += `<path d="M7 13 8 3l8 7zM33 13 32 3l-8 7z" fill="${b.body}"/>`;
    if (b.crest) x += `<path d="M17 10c0-5 3-8 8-8-2 2-2 5-1 8z" fill="${b.crest}"/>`;
    x += `<path d="M15 35v4M25 35v4" stroke="${b.feet}" stroke-width="2.4" stroke-linecap="round"/>`;
    x += `<rect x="6" y="8" width="28" height="28" rx="9" fill="${b.body}"/>`;
    x += `<ellipse cx="20" cy="28.5" rx="9.5" ry="6.5" fill="${b.belly}"/>`;
    if (b.wings) x += `<rect x="6.5" y="19" width="4" height="10" rx="2" fill="${b.wings}"/><rect x="29.5" y="19" width="4" height="10" rx="2" fill="${b.wings}"/>`;
    const r = b.big ? 4.4 : 3.2, pr = b.big ? 2.2 : 1.7;
    if (b.big) x += `<circle cx="14" cy="17.5" r="${r + 1.3}" fill="#F2A33A"/><circle cx="26" cy="17.5" r="${r + 1.3}" fill="#F2A33A"/>`;
    x += `<circle cx="14" cy="17.5" r="${r}" fill="#fff"/><circle cx="26" cy="17.5" r="${r}" fill="#fff"/>`;
    x += `<circle cx="14.6" cy="18" r="${pr}" fill="#16161a"/><circle cx="26.6" cy="18" r="${pr}" fill="#16161a"/>`;
    x += {
      small: `<path d="M17.4 21.6h5.2L20 25.8z" fill="${b.beakc}"/>`,
      thin: `<path d="M18.6 21.4h2.8L20 27.6z" fill="${b.beakc}"/>`,
      duck: `<ellipse cx="20" cy="23.4" rx="5.6" ry="2.7" fill="${b.beakc}"/>`,
      hook: `<path d="M16.6 21.2h6.8c.4 3.4-1.4 5.8-3.6 6.6.6-1.6.2-3-3.2-3.6z" fill="${b.beakc}"/>`,
      toucan: `<path d="M18.4 20.2H33c3 0 4.6 1.6 4.6 2.6-3.4 3.6-10 4-18.4 3.4z" fill="#FF8A1F"/><path d="M33.6 20.4c2.6.3 4 1.4 4 2.4-1 1-2.2 1.8-3.8 2.4z" fill="#222"/>`,
      flamingo: `<path d="M17.6 21.2h5.6c.2 2.6-.8 4.6-2.6 6l-1.6-2.6z" fill="#F7F1EA"/><path d="M21.6 25.6l-1 1.6-1.2-1.8z" fill="#222"/>`,
    }[b.beak];
    return `<symbol id="b${s}" viewBox="0 0 40 40">${x}</symbol>`;
  }
  // Agrega los <symbol id="b0".."b7"> a un <svg> oculto de la página.
  function install(svg) { svg.innerHTML = BIRD.map((_, s) => birdSymbol(s)).join(''); }
  root.CuBirdsArt = { BIRD, birdSymbol, install };
})(typeof self !== 'undefined' ? self : this);
