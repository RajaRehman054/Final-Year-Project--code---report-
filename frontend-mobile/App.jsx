import { useEffect } from 'react';
import messaging from '@react-native-firebase/messaging';
import {
	notificationListener,
	requestUserPermission,
} from './utils/pushNotifications.js';
import AppNav from './auth/AppNav.jsx';
import { AuthProvider } from './auth/AuthContext.jsx';
import SplashScreen from 'react-native-splash-screen';
import { Alert } from 'react-native';

const App = () => {
	useEffect(() => {
		SplashScreen.hide();
		requestUserPermission();
		notificationListener();
		const unsubscribe = messaging().onMessage(async remoteMessage => {
			Alert.alert(remoteMessage.notification.body);
		});
		return unsubscribe;
	}, []);

	return (
		<AuthProvider>
			<AppNav />
		</AuthProvider>
	);
};

export default App;
