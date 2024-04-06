/** @type {import('tailwindcss').Config} */
export default {
	content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
	theme: {
		extend: {
			colors: {
				background: '#F9F9F9',
				sidebar: '#FFFFFF',
				primary: '#FDA800',
				secondary: '#FFF2D9',
				heading: '#00261C',
				text: '#48525B',
				hover: '#FFF2D9',
			},
			boxShadow: {
				input: '0px 4px 10px 0px rgba(0, 0, 0, 0.10)',
			},
		},
		fontFamily: {
			manrope: ['Manrope', 'sans-serif'],
		},
	},
	plugins: [require('daisyui')],
};
