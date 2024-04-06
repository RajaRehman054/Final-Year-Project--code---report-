var express = require('express');
var router = express.Router();
var authenticate = require('../middleware/auth');
var contentManage = require('../controllers/contentManage');
var videoHandler = require('../middleware/videoHandler');
var thumbnail = require('../middleware/generateThumnail');

// ? Content Management Routes //
router.post(
	'/upload/',
	authenticate.verifyUser,
	contentManage.createVideo,
	videoHandler.upload.single('video'),
	thumbnail.generateThumbnail,
	contentManage.uploadVideo
);

router.get('/stream', contentManage.getVideo);
router.get('/', authenticate.verifyUser, contentManage.getVideos);
router.get('/myvideos', authenticate.verifyUser, contentManage.getMyVideos);
router.get('/other/:id', authenticate.verifyUser, contentManage.getOtherVideos);
router.get('/thumbnails', contentManage.getMyThumbnails);
router.get(
	'/followers',
	authenticate.verifyUser,
	contentManage.getFollowersVideos
);

module.exports = router;
