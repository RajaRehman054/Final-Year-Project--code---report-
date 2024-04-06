import React from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

import Login from '../screens/profiling/Login';
import Register from '../screens/profiling/Register';
import ForgotPassword from '../screens/profiling/ForgotPassword';
import EmailVerify from '../screens/profiling/EmailVerify';
import ChangePassword from '../screens/profiling/ChangePassword';

const Stack = createNativeStackNavigator();

const AuthStack = () => {
	return (
		<Stack.Navigator screenOptions={{ headerShown: false }}>
			<Stack.Screen name='Login' component={Login} />
			<Stack.Screen name='Register' component={Register} />
			<Stack.Screen name='ForgotPassword' component={ForgotPassword} />
			<Stack.Screen name='EmailVerify' component={EmailVerify} />
			<Stack.Screen name='ChangePassword' component={ChangePassword} />
		</Stack.Navigator>
	);
};

export default AuthStack;
