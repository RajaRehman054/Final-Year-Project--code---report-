var express = require('express');
var bodyParser = require('body-parser');
var router = express.Router();
router.use(bodyParser.json());
var fs = require('fs');
var path = require('path');

var asyncHandler = require('../middleware/asyncHandler');
var Video = require('../models/video');
var Bid = require('../models/bidding');
var User = require('../models/user');

exports.createVideo = async (req, res, next) => {
	const video = await Video.create({
		owner: req.user.hirer !== null ? req.user.hirer : req.user._id,
		description: req.query.desc,
		tags: req.headers.tags,
		bought: req.user.hirer !== null ? true : false,
	});
	await User.findByIdAndUpdate(req.user.id, {
		$inc: { videos_created: 1 },
	});
	req.video = video;
	next();
};

exports.uploadVideo = asyncHandler(async (req, res) => {
	await Video.findByIdAndUpdate(req.video._id, {
		path: req.source,
		thumbnail: req.thumbnail,
	});
	res.status(201).json();
});

exports.getVideo = (req, res, next) => {
	try {
		const videoPath = path.resolve(__dirname, `../${req.query.path}`);
		const videoSize = fs.statSync(videoPath).size;
		if (req.headers.range) {
			const range = req.headers.range;
			const chunksize = 1 * 1e6;
			const start = Number(range.replace(/\D/g, ''));
			const end = Math.min(start + chunksize, videoSize - 1);
			const contentLength = end - start + 1;
			const headers = {
				'Content-Range': `bytes ${start}-${end}/${videoSize}`,
				'Accept-Ranges': 'bytes',
				'Content-Length': contentLength,
				'Content-Type': 'video/mp4',
			};
			res.writeHead(206, headers);
			const stream = fs.createReadStream(videoPath, {
				start,
				end,
			});
			stream.pipe(res);
		} else {
			const headers = {
				'Content-Length': videoSize,
				'Content-Type': 'video/mp4',
			};
			res.writeHead(200, headers);
			fs.createReadStream(videoPath).pipe(res);
		}
	} catch (error) {
		next(error);
	}
};

exports.getVideos = asyncHandler(async (req, res) => {
	let ids = [];
	const bids = await Bid.find({ expired: false }).select('video');
	for (let index = 0; index < bids.length; index++) {
		ids.push(bids[index].video);
	}
	const videos = await Video.find({
		bought: false,
		_id: { $nin: ids },
		checked: true,
	}).populate('owner');
	res.status(201).json(videos);
});

exports.getFollowersVideos = asyncHandler(async (req, res) => {
	var array = [];
	for (let i = 0; i < req.user.following.length; i++) {
		const info = await User.findById(req.user.following[i].user);
		array.push(info._id);
	}
	const videos = await Video.find({
		owner: { $in: array },
		checked: true,
		bought: false,
	}).populate('owner');
	res.status(201).json(videos);
});

exports.getMyVideos = asyncHandler(async (req, res) => {
	const videos = await Video.find({ owner: req.user._id }).populate('owner');
	res.status(201).json(videos);
});

exports.getOtherVideos = asyncHandler(async (req, res) => {
	const videos = await Video.find({
		owner: req.params.id,
		checked: true,
	}).populate('owner');
	res.status(201).json(videos);
});

exports.getMyThumbnails = asyncHandler(async (req, res, next) => {
	res.sendFile(path.join(__dirname, `..${req.query.path}`), err => {
		if (err) {
			next(err);
		}
	});
});
