import { configureStore } from '@reduxjs/toolkit';
import userReducer from './userSlice';
import tokenReducer from './tokenSlice';

export const store = configureStore({
	reducer: {
		user: userReducer,
		token: tokenReducer,
	},
});
