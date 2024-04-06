import AsyncStorage from '@react-native-async-storage/async-storage';
import { useState, createContext, useEffect } from 'react';
import {removeFCM} from '../utils/pushNotifications';

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
	const [user, setUser] = useState(null);
	const [loading, setLoading] = useState(true);

	const logIn = async token => {
		setLoading(true);
		await AsyncStorage.setItem('user', token);
		setUser(token);
		setLoading(false);
	};
	const logOut = async () => {
		setLoading(true);
		await removeFCM();
		await AsyncStorage.removeItem('user');
		setUser(null);
		setLoading(false);
	};

	const isLoggedIn = async () => {
		try {
			const token = await AsyncStorage.getItem('user');
			setUser(token);
			setLoading(false);
		} catch (error) {
			console.log(`Error is ${error}.`);
		}
	};

	useEffect(() => {
		isLoggedIn();
	});

	return (
		<AuthContext.Provider value={{ user, loading, logIn, logOut }}>
			{children}
		</AuthContext.Provider>
	);
};
