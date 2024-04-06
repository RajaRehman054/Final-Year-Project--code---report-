import { createSlice } from '@reduxjs/toolkit';

const userSlice = createSlice({
	name: 'user',
	initialState: { user: {} },
	reducers: {
		setUser(state, action) {
			state.user = action.payload;
		},
		clearUser(state, action) {
			state.user = { user: {} };
		},
	},
});

export const { setUser, clearUser } = userSlice.actions;
export default userSlice.reducer;
