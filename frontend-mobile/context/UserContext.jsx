import { useState, createContext, useEffect, useContext } from 'react';
import { AuthContext } from '../auth/AuthContext';
import url from '../utils/url';

export const UserContext = createContext();

export const UserProvider = ({ children }) => {
	const { user } = useContext(AuthContext);
	const [loading, setLoading] = useState(true);
	const [userData, setUserData] = useState(null);

	const getUser = async () => {
		const res = await fetch(`${url}/users/user`, {
			headers: {
				'content-type': 'application/json',
				Authorization: `Bearer ${user}`,
			},
		});
		const response = await res.json();
		setUserData(response);
		setLoading(false);
	};

	useEffect(() => {
		getUser();
	}, []);

	return (
		<UserContext.Provider value={{ userData, loading, getUser }}>
			{children}
		</UserContext.Provider>
	);
};
