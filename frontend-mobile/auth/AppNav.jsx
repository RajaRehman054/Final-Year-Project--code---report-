import { useContext } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import Toast from 'react-native-toast-message';

import AuthStack from './AuthStack';
import MainStack from './MainStack';
import { AuthContext } from './AuthContext';
import Loader from '../components/Loader';
import { UserProvider } from '../context/UserContext';

const AppNav = () => {
	const { loading, user } = useContext(AuthContext);

	if (loading) {
		return <Loader />;
	}
	return (
		<>
			<NavigationContainer>
				{user !== null ? (
					<UserProvider>
						<MainStack />
					</UserProvider>
				) : (
					<AuthStack />
				)}
			</NavigationContainer>
			<Toast />
			<Toast />
		</>
	);
};

export default AppNav;
