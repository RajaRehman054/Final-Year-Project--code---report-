import { createSlice } from '@reduxjs/toolkit';

const tokenSlice = createSlice({
	name: 'token',
	initialState: { token: null },
	reducers: {
		setToken(state, action) {
			state.token = action.payload;
		},
		clearToken(state, action) {
			state.token = { toen: null };
		},
	},
});

export const { setToken, clearToken } = tokenSlice.actions;
export default tokenSlice.reducer;
