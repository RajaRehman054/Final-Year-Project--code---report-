import Pic from '../../assets/img1.png';
import { useLocation } from 'react-router-dom';
import ContentdetailsModal from '../../Components/BidComp/contentdetailsModal';
import { HOSTNAME } from '../../../Config';
import { format } from 'timeago.js';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import BuyNowButton from '../../Components/BidComp/BuyNowButton';

const ContentDetails = () => {
	const { state } = useLocation();
	let element = state;

	const handleModal = async checker => {
		if (checker?.bubbles === true) {
			return document.getElementById('my_modal_3').showModal();
		}
		document.getElementById('my_modal_3').close();
	};

	return (
		<div className='flex w-full flex-col'>
			{/* Top Area */}
			<div className='flex w-full flex-col justify-between'>
				<h1 className='text-[20px] font-bold'>Content Details</h1>
			</div>
			<div className='my-[22px] flex h-[300px] w-full flex-row flex-wrap  rounded-xl bg-[white] shadow-xl '>
				<video
					className='h-[300px] w-full rounded-xl  bg-[white] '
					controls
					controlsList='nodownload'
					muted>
					<source
						src={`${HOSTNAME}videos/stream?path=${element.video.path}`}
						type='video/mp4'
					/>
					Your browser does not support the video tag.
				</video>
			</div>
			<div className='my-[22px] flex  w-full flex-col flex-wrap  rounded-xl bg-[white] shadow-xl '>
				<div className=' mb-[10px]  flex h-[80px] cursor-pointer flex-row items-center justify-between  bg-[white] p-4 '>
					<div className='avatar items-center'>
						<div className='w-[50px] rounded-full'>
							<img
								src={
									element.original_owner.picture !== null
										? `${HOSTNAME}users/picture?path=${element.original_owner.picture}`
										: Pic
								}
							/>
						</div>
					</div>

					<div className='mr-[auto]  px-5'>
						<p className='text-[14px] font-bold text-[black] sm:text-[24px]'>
							{element.original_owner.username}
							<span className='ml-[10px] text-[12px] text-[black]'>
								{element.original_owner.email}
							</span>
						</p>
						<p className='text-[12px] text-[black]'>
							{format(element.video.created_on)}
						</p>
					</div>

					<div className=' flex flex-row items-center justify-between '>
						<button
							className='mx-[10px] h-[22px] w-[100px] justify-center rounded-[10px] bg-[#FF8216]  text-[15px]  text-white hover:bg-[#F7A541] active:bg-[#FFD583] sm:h-[35px] sm:w-[200px]'
							style={{ transition: '0.3s' }}
							onClick={handleModal}>
							Place a Bid
						</button>
						<BuyNowButton videoItem={element} />
					</div>
				</div>

				<div className='p-4'>
					<p className='text-[14px]'>{element.video.descp}</p>
				</div>
				<div className='bg-[#D9D9D9]  p-1'>
					<p className='ml-[4px] text-[22px] font-bold text-[#4E4E4E]'>
						Content details
					</p>
				</div>

				<div className='m-[22px] w-[full] overflow-x-auto'>
					<table className='w-full table-auto rounded-xl text-[12px] font-normal text-text  md:text-[14px]'>
						<tbody>
							<tr className='cursor-pointer rounded-xl  bg-white hover:bg-[#EBEBEB] '>
								<td className='font-medium'>
									<p>Captured On</p>
								</td>
								<td className='font-medium'>
									<p className='pr-1'>Mobile</p>
								</td>
							</tr>
							<tr className='cursor-pointer rounded-xl  bg-white hover:bg-[#EBEBEB] '>
								<td className='font-medium'>
									<p>City</p>
								</td>
								<td className='font-medium'>
									<p className='pr-1'>Rawalpindi</p>
								</td>
							</tr>
							<tr className='cursor-pointer rounded-xl  bg-white hover:bg-[#EBEBEB] '>
								<td className='font-medium'>
									<p>Country</p>
								</td>
								<td className='font-medium'>
									<p className='pr-1'>Pakistan</p>
								</td>
							</tr>
							<tr className='cursor-pointer rounded-xl  bg-white hover:bg-[#EBEBEB] '>
								<td className='font-medium'>
									<p>Category</p>
								</td>
								<td className='font-medium'>
									<p className='pr-1'>Video</p>
								</td>
							</tr>
							<tr className='cursor-pointer rounded-xl  bg-white hover:bg-[#EBEBEB] '>
								<td className='font-medium'>
									<p>Video Ratio</p>
								</td>
								<td className='font-medium'>
									<p className='pr-1'>16:9</p>
								</td>
							</tr>
						</tbody>
					</table>
				</div>

				<div className='bg-[#D9D9D9]  p-1'>
					<p className='ml-[4px] text-[22px] font-bold text-[#4E4E4E]'>
						Highest Bids
					</p>
				</div>
			</div>
			<div className='flex flex-wrap justify-between'>
				{element.list.map(elem => {
					return (
						<div className=' mb-[10px]  flex h-[80px] w-[48%] cursor-pointer flex-row items-center justify-between  rounded-[20px] bg-[white]  p-4  hover:bg-[#EBEBEB]'>
							<div className='avatar items-center'>
								<div className='w-[50px] rounded-full'>
									<img
										src={
											elem.user.picture !== null
												? `${HOSTNAME}users/picture?path=${elem.user.picture}`
												: Pic
										}
									/>
								</div>
							</div>
							<div className='mr-[auto] px-5'>
								<p className='font-bold text-[black]'>
									{elem.user.username}
								</p>
							</div>
							<div className='h-[50px] w-[100px] justify-center  text-[24px] '>
								<p className='text-center'>{elem.amount}</p>
							</div>
						</div>
					);
				})}
			</div>
			{/*Modal */}
			<ContentdetailsModal data={element} handleModal={handleModal} />
			<ToastContainer />
		</div>
	);
};

export default ContentDetails;
