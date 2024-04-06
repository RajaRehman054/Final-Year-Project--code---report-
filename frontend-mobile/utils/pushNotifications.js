import messaging from '@react-native-firebase/messaging';

export const getFCM = async () => {
	try {
		const fcmToken = await messaging().getToken();
		return fcmToken;
	} catch (error) {
		console.log(error);
	}
};

export const removeFCM = async () => {
	try {
		const fcmToken = await messaging().deleteToken();
		return fcmToken;
	} catch (error) {
		console.log(error);
	}
};


export const requestUserPermission = async () => {
	const authStatus = await messaging().requestPermission();
	const enabled =
		authStatus === messaging.AuthorizationStatus.AUTHORIZED ||
		authStatus === messaging.AuthorizationStatus.PROVISIONAL;

	if (enabled) {
		console.log('Authorization status:', authStatus);
	}
};

export const notificationListener = () => {
	messaging().onNotificationOpenedApp(remoteMessage => {
		console.log('Notification caused app to open from background state.');
	});
	messaging()
		.getInitialNotification()
		.then(remoteMessage => {
			if (remoteMessage) {
				console.log('Notification caused app to open from quit state.');
			}
		});
};
