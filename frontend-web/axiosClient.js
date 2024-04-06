import axios from 'axios';
import { HOSTNAME } from './Config';
import { store } from './src/redux/store';

const axiosClient = axios.create({
	baseURL: HOSTNAME,
	headers: {
		'Content-Type': 'application/json',
	},
});

// Add a request interceptor to pass the token with each request
axiosClient.interceptors.request.use(async config => {
	const token = store.getState()?.token?.token || localStorage.getItem('jwt');
	if (token) {
		config.headers.Authorization = `Bearer ${token}`;
	}
	return config;
});

export default axiosClient;
