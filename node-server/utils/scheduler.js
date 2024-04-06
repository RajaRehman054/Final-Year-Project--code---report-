const cron = require('node-cron');
const pushNotification = require('../utils/pushNotifications');
const axios = require('axios');
const FormData = require('form-data');
const path = require('path');
const fs = require('fs');

const Bid = require('../models/bidding');
const Video = require('../models/video');
const User = require('../models/user');
const Posting = require('../models/posting');
const Notification = require('../models/notification');

const notification = require('../utils/pushNotifications');

exports.expireBid = async () => {
	cron.schedule('*/10 * * * * **', async () => {
		const totalBids = await Bid.find({
			expired: false,
		});

		totalBids.forEach(async document => {
			let currentTime = Date.now();
			if (document.expires < currentTime) {
				await Bid.findByIdAndUpdate(document._id, { expired: true });
				await User.findByIdAndUpdate(document.original_owner, {
					$inc: { wallet: document.current_amount, sales: 1 },
				});
				await User.findByIdAndUpdate(document.current_highest, {
					$inc: { bought: 1 },
				});
				await Video.findByIdAndUpdate(document.video, {
					owner: document.current_highest,
				});
				const user = await User.findById(document.original_owner);
				user.fcm !== null
					? await notification.sendBidExpired(
							document.current_amount,
							[user.fcm]
					  )
					: null;
			}
		});
	});
};

exports.jobChecker = async () => {
	cron.schedule('* * * * *', async () => {
		const postings = await Posting.find({ ended: false });

		postings.forEach(async document => {
			let currentTime = Date.now();
			if (document.time < currentTime) {
				await Posting.findByIdAndUpdate(document._id, {
					ended: true,
					receiving: false,
				});
				document.requests.forEach(async element => {
					await User.findByIdAndUpdate(element.user, {
						hirer: null,
					});
				});
			}
		});
	});
};

exports.sendRecommendations = async () => {
	cron.schedule('*/30 * * * * *', async () => {
		const threeHoursAgo = new Date();
		threeHoursAgo.setHours(threeHoursAgo.getHours() - 3);
		const users = await User.find({});
		const videos = await Video.find({
			bought: false,
			created_on: { $gte: threeHoursAgo },
		});
		if (videos.length > 5) {
			let array = [];
			users.forEach(user => {
				if (user.fcm !== '') {
					array.push(user.fcm);
				}
			});
			await pushNotification.recommendationNotification(array);
		}
	});
};

exports.filterUploadedVideos = async () => {
	cron.schedule('*/30 * * * * *', async () => {
		try {
			const videos = await Video.find({
				checked: false,
				checking: false,
			});
			videos.forEach(async document => {
				const formData = new FormData();
				const fileStream = fs.createReadStream(
					path.resolve(__dirname, `..${document.path}`)
				);
				await Video.findByIdAndUpdate(document._id, {
					checking: true,
				});
				formData.append('file', fileStream);
				const response = await axios.post(
					process.env.FLASK_SERVER,
					formData,
					{
						headers: {
							'Content-Type': `multipart/form-data`,
						},
					}
				);
				if (!response.data.notSafe) {
					const user = await User.findById(document.owner);
					await Video.findByIdAndUpdate(document._id, {
						checked: true,
					});
					!user.buyer
						? user.fcm !== null
							? await notification.videoFiltered(true, [user.fcm])
							: null
						: await Notification.create({
								user: user._id,
								message: `Your video ${document.description} has been successfully uploaded.`,
						  });
				} else {
					fs.unlink(
						path.resolve(__dirname, `..${document.path}`),
						err => {
							if (err) {
								console.log(err);
							}
						}
					);
					fs.unlink(
						path.resolve(__dirname, `..${document.thumbnail}`),
						err => {
							if (err) {
								console.log(err);
							}
						}
					);
					await Video.findByIdAndDelete(document._id);
					const user = await User.findById(document.owner);
					!user.buyer
						? user.fcm !== null
							? await notification.videoFiltered(false, [
									user.fcm,
							  ])
							: null
						: await Notification.create({
								user: user._id,
								message: `Your video ${document.description} has been removed due to obscene content.`,
						  });
				}
			});
		} catch (error) {
			console.log(error);
		}
	});
};
