import {
	IoHeart,
	IoChatbubble,
	IoEllipsisHorizontal,
	IoColorPalette,
	IoReloadCircle,
	IoMicOff,
	IoStarHalf,
} from 'react-icons/io5';
import { format } from 'timeago.js';
import UserOne from '../../assets/img1.png';
import { toast } from 'react-toastify';
import Axios from 'axios';
import FileDownload from 'js-file-download';
import { HOSTNAME } from '../../../Config';
import { store } from '../../redux/store';

const MyProfileVideos = ({ data }) => {
	const upscale = async path => {
		const token =
			store.getState()?.token?.token || localStorage.getItem('jwt');
		toast.success(`Download will begin shortly`, {
			position: 'bottom-right',
			autoClose: 1500,
			hideProgressBar: false,
			closeOnClick: true,
			pauseOnHover: true,
			draggable: true,
			progress: undefined,
			theme: 'dark',
		});
		const response = await Axios({
			url: `${HOSTNAME}api/video/enhance?path=${path}`,
			method: 'GET',
			headers: {
				Authorization: `Bearer ${token}`,
			},
			responseType: 'blob',
		});
		FileDownload(response.data, 'upscaled.mp4');
	};
	const rotate = async path => {
		const token =
			store.getState()?.token?.token || localStorage.getItem('jwt');
		toast.success(`Download will begin shortly`, {
			position: 'bottom-right',
			autoClose: 1500,
			hideProgressBar: false,
			closeOnClick: true,
			pauseOnHover: true,
			draggable: true,
			progress: undefined,
			theme: 'dark',
		});
		const response = await Axios({
			url: `${HOSTNAME}api/video/rotate?path=${path}`,
			method: 'GET',
			headers: {
				Authorization: `Bearer ${token}`,
			},
			responseType: 'blob',
		});
		FileDownload(response.data, 'rotate.mp4');
	};

	const noAudio = async path => {
		const token =
			store.getState()?.token?.token || localStorage.getItem('jwt');
		toast.success(`Download will begin shortly`, {
			position: 'bottom-right',
			autoClose: 1500,
			hideProgressBar: false,
			closeOnClick: true,
			pauseOnHover: true,
			draggable: true,
			progress: undefined,
			theme: 'dark',
		});
		const response = await Axios({
			url: `${HOSTNAME}api/video/noaudio?path=${path}`,
			method: 'GET',
			headers: {
				Authorization: `Bearer ${token}`,
			},
			responseType: 'blob',
		});
		FileDownload(response.data, 'noaudio.mp4');
	};

	const blackwhite = async path => {
		const token =
			store.getState()?.token?.token || localStorage.getItem('jwt');
		toast.success(`Download will begin shortly`, {
			position: 'bottom-right',
			autoClose: 1500,
			hideProgressBar: false,
			closeOnClick: true,
			pauseOnHover: true,
			draggable: true,
			progress: undefined,
			theme: 'dark',
		});
		const response = await Axios({
			url: `${HOSTNAME}api/video/blackwhite?path=${path}`,
			method: 'GET',
			headers: {
				Authorization: `Bearer ${token}`,
			},
			responseType: 'blob',
		});
		FileDownload(response.data, 'blackwhite.mp4');
	};
	return (
		<>
			{data.map((element, index) => (
				<div
					key={index}
					className='card card-compact mb-[22px] w-[85%] bg-white shadow-xl'>
					<video
						className='h-[300px] w-full rounded-xl  bg-[white] '
						controls
						controlsList='nodownload'
						muted
						autoPlay>
						<source
							src={`${HOSTNAME}videos/stream?path=${element.path}`}
							type='video/mp4'
						/>
						Your browser does not support the video tag.
					</video>
					<div className='card-body'>
						<div className='flex flex-row items-center '>
							<div className='avatar items-center'>
								<div className='w-[50px] rounded-full'>
									<img
										src={
											element.owner.picture
												? `${HOSTNAME}users/picture?path=${element.owner.picture}`
												: UserOne
										}
									/>
								</div>
							</div>

							<div className='mr-[auto]  px-5'>
								<p className='text-[14px] font-bold text-[black] '>
									{data.name}
									<span className='ml-[10px] text-[12px] text-[black]'>
										{element.owner.username}
									</span>
								</p>
								<p className='text-[12px] text-[black]'>
									{format(element.created_on)}
								</p>
							</div>

							<div className='ml-[auto] cursor-pointer'>
								<IoEllipsisHorizontal
									size={30}
									color='#939393'
								/>
							</div>
						</div>
						<p className='font-semibold'>{element.description}</p>
						<p className='font-semibold text-[#FF8216]'>
							{element.tags}
						</p>
						<div className='card-actions items-center justify-between'>
							<div className='my-2 flex items-center'>
								<div className='mr-5 flex cursor-pointer flex-row '>
									<IoHeart size={24} color='#939393' />
									<p>{element.likes || 0}</p>
								</div>
								<div className='flex cursor-pointer flex-row '>
									<IoChatbubble size={24} color='#939393' />
									<p>{element.comments || 9}</p>
								</div>
							</div>

							<div className='my-2 flex cursor-pointer items-center  '>
								<div
									className='lg:tooltip mr-5'
									data-tip='No Audio'
									onClick={() => noAudio(element.path)}>
									<IoMicOff size={24} color='#939393' />
								</div>
								<div
									className='lg:tooltip mr-5'
									data-tip='Black & White'
									onClick={() => blackwhite(element.path)}>
									<IoStarHalf size={24} color='#939393' />
								</div>
								<div className='lg:tooltip mr-5' data-tip='90°'>
									<IoReloadCircle
										size={24}
										color='#939393'
										onClick={() => rotate(element.path)}
									/>
								</div>
								<div
									className='lg:tooltip'
									data-tip='Enhancement'>
									<IoColorPalette
										size={24}
										color='#939393'
										onClick={() => upscale(element.path)}
									/>
								</div>
							</div>
						</div>
					</div>
				</div>
			))}
		</>
	);
};

export default MyProfileVideos;
