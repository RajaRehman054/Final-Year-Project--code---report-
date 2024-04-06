import { IoHeart, IoChatbubble, IoDownload } from 'react-icons/io5';
import { HOSTNAME } from '../../../Config';
import { format } from 'timeago.js';

import MakeabidModal from './MakeabidModal';

const Following = ({ videos }) => {
	const handleModal = async checker => {
		if (checker?.bubbles === true) {
			return document.getElementById('my_modal_3').showModal();
		}
		document.getElementById('my_modal_3').close();
	};

	return (
		<>
			{videos.map(element => (
				<div className='card card-compact mb-[22px] w-[85%] bg-white shadow-xl'>
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
										src={`${HOSTNAME}users/picture?path=${element.owner.picture}`}
									/>
								</div>
							</div>

							<div className='mr-[auto]  px-5'>
								<p className='text-[14px] font-bold text-[black] '>
									{element.owner.username}
									<span className='ml-[10px] text-[12px] text-[black]'>
										{element.owner.email}
									</span>
								</p>
								<p className='text-[12px] text-[black]'>
									{format(element.created_on)}
								</p>
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
									className='lg:tooltip mr-2'
									data-tip='Make a Bid'
									onClick={handleModal}>
									<IoDownload size={24} color='#939393' />
								</div>
							</div>
						</div>
					</div>
					<MakeabidModal showModal={handleModal} id={element._id} />
				</div>
			))}
		</>
	);
};

export default Following;
