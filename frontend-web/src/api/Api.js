import axiosClient from '../../axiosClient';

//TODO profiling
export const loginApi = async data => {
	try {
		const response = await axiosClient.post('api/login', data);
		return response.data;
	} catch (error) {
		return error.response.data;
	}
};

export const registerApi = async data => {
	try {
		const response = await axiosClient.post('api/register', data);
		return response.data;
	} catch (error) {
		return error.response.data;
	}
};

export const otpGenerateApi = async email => {
	try {
		const response = await axiosClient.get(`users/otp/${email}`);
		return response.data;
	} catch (error) {
		return error.response.data;
	}
};

export const verifyOtp = async data => {
	try {
		const response = await axiosClient.get(
			`users/otpVerify/${data.email}/${data.otp}`
		);
		return response.data;
	} catch (error) {
		return error.response.data;
	}
};

export const resetPasswordApi = async data => {
	try {
		const response = await axiosClient.patch(`users/passwordreset`, data);
		return response.data;
	} catch (error) {
		return error.response.data;
	}
};

export const profilePicture = async data => {
	try {
		const response = await axiosClient.patch(`users/profilepicture/web`, {
			picture: data,
		});
		return response.data;
	} catch (error) {
		return error.response.data;
	}
};

export const getUser = async () => {
	const response = await axiosClient.get(`users/user`);
	return response.data;
};

export const editUser = async payload => {
	try {
		const response = await axiosClient.patch(`users/profile/edit`, payload);
		return response.data;
	} catch (error) {
		return error.response.data;
	}
};

export const changePass = async payload => {
	try {
		const response = await axiosClient.patch(
			`users/profile/password/change`,
			payload
		);
		return response.data;
	} catch (error) {
		return error.response.data;
	}
};

export const addFollower = async id => {
	const response = await axiosClient.patch(`users/follower/add/${id}`);
	return response.data;
};

export const removeFollower = async id => {
	const response = await axiosClient.patch(`users/follower/remove/${id}`);
	return response.data;
};

//TODO video apis
export const getVideosApi = async () => {
	const response = await axiosClient.get('videos');
	return response.data;
};

export const getOtherVideosApi = async id => {
	const response = await axiosClient.get(`videos/other/${id}`);
	return response.data;
};

export const getFollowingVideosApi = async () => {
	const response = await axiosClient.get('videos/followers');
	return response.data;
};

export const getFeedData = async () => {
	const response = await axiosClient.get('api/feed/data');
	return response.data;
};

export const getMyVideosApi = async () => {
	const response = await axiosClient.get('videos/myvideos');
	return response.data;
};

//TODO bids api
export const createBid = async payload => {
	const response = await axiosClient.post('api/bid/create', payload);
	return response.data;
};

export const getBidsApi = async () => {
	const response = await axiosClient.get('api/bid/all');
	return response.data;
};

export const getinvolvedBidsApi = async () => {
	const response = await axiosClient.get('api/bid/involved');
	return response.data;
};

export const updateBid = async (id, payload) => {
	try {
		const response = await axiosClient.patch(
			`api/bid/update/${id}`,
			payload
		);
		return response.data;
	} catch (error) {
		return error.response.data;
	}
};

//TODO jobs api
export const getJobs = async () => {
	const response = await axiosClient.get(`api/jobs`);
	return response.data;
};

export const getPastJobs = async () => {
	const response = await axiosClient.get(`api/jobs/past`);
	return response.data;
};

export const createJob = async data => {
	const response = await axiosClient.post(`api/job/create`, data);
	return response.data;
};

export const stopRequests = async id => {
	const response = await axiosClient.patch(`api/job/update/${id}`);
	return response.data;
};

export const acceptRejectRequests = async (reject, id, sid) => {
	if (reject) {
		const response = await axiosClient.patch(
			`api/job/accept/${id}/${sid}?reject=true`
		);
		return response.data;
	} else {
		try {
			const response = await axiosClient.patch(
				`api/job/accept/${id}/${sid}`
			);
			return response.data;
		} catch (error) {
			return error.response.data;
		}
	}
};

export const deleteJob = async id => {
	try {
		const response = await axiosClient.delete(`api/job/delete/${id}`);
		return response.data;
	} catch (error) {
		return error.response.data;
	}
};

//TODO : payment
export const payment = async payload => {
	try {
		const response = await axiosClient.post(`payment/intent/${payload}`);
		return response.data;
	} catch (error) {
		return error.response.data;
	}
};

export const paymentHistory = async payload => {
	try {
		const response = await axiosClient.get(`payment/history`);
		return response.data;
	} catch (error) {
		return error.response.data;
	}
};

export const getNotifications = async payload => {
	try {
		const response = await axiosClient.get(`api/notifications`);
		return response.data;
	} catch (error) {
		return error.response.data;
	}
};

export const changeNotification = async payload => {
	const response = await axiosClient.patch(
		`api/notification/setting`,
		payload
	);
	return response.data;
};

//TODO: search Videos
export const searchVideosApi = async (type, value) => {
	const response = await axiosClient.get(
		`api/videos?type=${type}&value=${value}`
	);
	return response.data;
};
