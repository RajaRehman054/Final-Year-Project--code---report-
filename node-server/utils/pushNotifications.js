require('dotenv').config();
var FCM = require('fcm-node');
var serverKey = process.env.FCM;
var fcm = new FCM(serverKey);

exports.startedBidNotification = async (ids, message) => {
	try {
		var pushMessage = {
			registration_ids: ids,
			content_available: true,
			mutable_content: true,
			notification: {
				title: 'Video Update',
				body: 'You have an active bid on one of your videos.',
				icon: 'myicon',
				sound: 'mySound',
			},
		};

		fcm.send(pushMessage, (err, response) => {
			if (err) {
				console.log('Something has gone wrong!', err);
			} else {
				console.log('Successfully sent with response: ', response);
			}
		});
	} catch (error) {
		console.log('Error is:', error);
	}
};

exports.recommendationNotification = async ids => {
	try {
		var pushMessage = {
			registration_ids: ids,
			content_available: true,
			mutable_content: true,
			notification: {
				title: 'Something Happening!',
				body: 'Go out and record something is going on.',
				icon: 'myicon',
				sound: 'mySound',
			},
		};

		fcm.send(pushMessage, (err, response) => {
			if (err) {
				console.log('Something has gone wrong!', err);
			} else {
				console.log('Successfully sent with response: ', response);
			}
		});
	} catch (error) {
		console.log('Error is:', error);
	}
};

exports.videoFiltered = async (safe, ids) => {
	try {
		var pushMessage = {
			registration_ids: ids,
			content_available: true,
			mutable_content: true,
			notification: {
				title: 'Video Update',
				body: !safe
					? 'Your video contains obscene content, hence it has been removed.'
					: 'Your video has been successfully uploaded.',
				icon: 'myicon',
				sound: 'mySound',
			},
		};

		fcm.send(pushMessage, (err, response) => {
			if (err) {
				console.log('Something has gone wrong!', err);
			} else {
				console.log('Successfully sent with response: ', response);
			}
		});
	} catch (error) {
		console.log('Error is:', error);
	}
};

exports.acceptedJobRequest = async (reject, job, ids) => {
	try {
		var pushMessage = {
			registration_ids: ids,
			content_available: true,
			mutable_content: true,
			notification: {
				title: reject ? 'Important! ' : 'Congratulations!',
				body: reject
					? `Your job request for ${job} has been rejected.`
					: `Your job request for ${job} has been accepted.`,
				icon: 'myicon',
				sound: 'mySound',
			},
		};

		fcm.send(pushMessage, (err, response) => {
			if (err) {
				console.log('Something has gone wrong!', err);
			} else {
				console.log('Successfully sent with response: ', response);
			}
		});
	} catch (error) {
		console.log('Error is:', error);
	}
};

exports.sendBidExpired = async (money, ids) => {
	try {
		var pushMessage = {
			registration_ids: ids,
			content_available: true,
			mutable_content: true,
			notification: {
				title: 'Video Sold!',
				body: `Your video has been sold for ${money}`,
				icon: 'myicon',
				sound: 'mySound',
			},
		};

		fcm.send(pushMessage, (err, response) => {
			if (err) {
				console.log('Something has gone wrong!', err);
			} else {
				console.log('Successfully sent with response: ', response);
			}
		});
	} catch (error) {
		console.log('Error is:', error);
	}
};
