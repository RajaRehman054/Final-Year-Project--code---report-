import UserOne from '../../assets/img1.png';
import { useNavigate } from 'react-router-dom';
import { HOSTNAME } from '../../../Config';

const BidsModal = ({ showmodal, data, channels, showContent }) => {
	const navigate = useNavigate();
	const arrayElement = showContent === 2 ? data : channels;
	const heading = 'Top Creators';
	const heading1 = 'Top Channels';

	return (
		<dialog id='my_modal' className='modal '>
			<div className='modal-box w-[400px] bg-white'>
				<form method='dialog'>
					<button
						className='btn btn-circle btn-ghost btn-sm absolute right-2 top-2'
						onClick={showmodal}>
						✕
					</button>
				</form>
				<div>
					<h3 className='text-center text-lg font-bold text-[#FF8216]'>
						{showContent === 2 ? heading : heading1}
					</h3>
				</div>
				{arrayElement.map((element, index) => (
					<div
						key={index}
						className=' mb-[10px]  flex h-[80px] cursor-pointer flex-row items-center justify-between  rounded-[20px] bg-[white]  p-4  hover:bg-[#EBEBEB]'
						onClick={() =>
							navigate(`/otherProfile`, {
								state: element,
							})
						}>
						<div className='avatar items-center'>
							<div className='w-[50px] rounded-full'>
								<img
									src={
										element.picture !== null
											? `${HOSTNAME}users/picture?path=${element.picture}`
											: UserOne
									}
								/>
							</div>
						</div>
						<div className='mr-[auto] px-5'>
							<p className='font-bold text-[black]'>
								{element.username}
							</p>
							<p className='text-[black]'>
								{element.buyer
									? element.bought || 0
									: element.created_videos || 0}
								<span> items</span>
							</p>
						</div>
						<button
							className='h-[22px] w-[100px] justify-center rounded-[10px] border-[1px]  border-[#FF8216] text-[15px] text-[black] hover:bg-[#FF8216] hover:text-white active:bg-[#FFD583]'
							style={{ transition: '0.3s' }}>
							Follow
						</button>
					</div>
				))}
			</div>
		</dialog>
	);
};

export default BidsModal;
