import { IoTrashOutline } from 'react-icons/io5';
import { useLocation, useNavigate } from 'react-router-dom';
import { HOSTNAME } from '../../../Config';
import { format } from 'timeago.js';
import { ToastContainer, toast } from 'react-toastify';
import { useSelector } from 'react-redux';
import Heading from '../../Components/Heading';
import { acceptRejectRequests, deleteJob, stopRequests } from '../../api/Api';
import { useState } from 'react';
import { useDispatch } from 'react-redux';
import { setUser } from '../../redux/userSlice';
import { getUser } from '../../api/Api';

const JobDetails = ({}) => {
	let { state } = useLocation();
	const navigate = useNavigate();
	const { user } = useSelector(state => state.user);
	const [receiving, setReceiving] = useState(state.receiving);
	const dispatch = useDispatch();
	const fetchUsers = async () => {
		const data = await getUser();
		dispatch(setUser(data.user));
	};
	const stopRequest = async () => {
		await stopRequests(state._id);
		setReceiving(prev => !prev);
	};
	const acceptRequest = async (id, sid) => {
		const res = await acceptRejectRequests(false, id, sid);
		if (res.message === 'Not enough balance.') {
			return toast.error(`${res.message}`, {
				position: 'bottom-right',
				autoClose: 5000,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		await fetchUsers();
		navigate('/');
	};
	const rejectRequest = async (id, sid) => {
		await acceptRejectRequests(true, id, sid);
		navigate('/');
	};
	const deleteJobFunc = async () => {
		const res = await deleteJob(state._id);
		if (
			res.message ===
			'Job cannot be deleted as there is an accepted request.'
		) {
			return toast.error(`${res.message}`, {
				position: 'bottom-right',
				autoClose: 5000,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		}
		navigate('/');
	};

	if (!state.past) {
		return (
			<div className='flex w-full flex-col'>
				{/* Top Area */}
				<div className='flex w-full flex-col justify-between'>
					<h1 className='text-[20px] font-bold'>Job Details</h1>
				</div>

				{/* Description Card */}
				<div className='my-[22px] flex  w-full flex-row flex-wrap  rounded-xl bg-[white] shadow-xl '>
					<div className=' mb-[10px]  flex h-[80px] w-full  flex-row items-center justify-between rounded-xl  bg-[white] p-4 '>
						<div className='avatar items-center'>
							<div className='w-[50px] rounded-full'>
								<img
									src={
										state.creator.picture !== null
											? `${HOSTNAME}users/picture?path=${state.creator.picture}`
											: UserOne
									}
								/>
							</div>
						</div>

						<div className='mr-[auto]  px-5'>
							<p className='text-[14px] font-bold text-[black] sm:text-[24px]'>
								{state.creator.username}
								<span className='ml-[10px] text-[12px] text-[black]'>
									{state.creator.email}
								</span>
							</p>
							<p className='text-[12px] text-[black]'>
								{state.expiry_date}
							</p>
						</div>

						{/* Buttons */}
						<div className=' ml-[auto] flex flex-row items-center justify-between '>
							<button
								className='mx-[10px] h-[22px] w-[100px] justify-center rounded-[10px] bg-[#FF8216]  text-[15px]  text-white hover:bg-[#F7A541] active:bg-[#FFD583] sm:h-[35px] sm:w-[200px]'
								style={{ transition: '0.3s' }}
								onClick={stopRequest}>
								{receiving
									? 'Stop Receving'
									: 'Start Receiving'}
							</button>
							<button
								className='h-[22px] w-[100px] justify-center rounded-[10px] bg-[#FF8216]  text-[15px]   text-white hover:bg-[#F7A541] active:bg-[#FFD583] sm:h-[35px] sm:w-[200px]'
								style={{ transition: '0.3s' }}
								onClick={deleteJobFunc}>
								Delete Job
							</button>
						</div>
					</div>

					<div className='mb-[20px]  flex w-full flex-row justify-between rounded-xl bg-[white] px-4'>
						<div className='w-[80%]'>
							<p className='text-[14px] text-[black] '>
								{state.details}
							</p>
							<p className='text-[12px] text-[black]'>
								Posted On:
								<span> {format(state.createdAt)}</span>
							</p>
						</div>
						<div className='w-[20%] flex items-center justify-center'>
							<p className=' text-[24px] font-bold text-[black]'>
								{state.price}
							</p>
						</div>
					</div>
				</div>

				{/*Heading */}
				<Heading text={'Job Candidates'} />

				{/* Job Candidates Table */}
				<div className='my-[22px] w-[full] overflow-x-auto'>
					<table className='w-full table-auto rounded-xl bg-[white] text-[12px] font-normal text-text  md:text-[14px]'>
						<thead>
							<tr className='border-b-[3px] border-[#E4E4E4]'>
								<th className='pb-[8px] pl-4 pt-[18px] text-left'>
									Picture
								</th>
								<th className='pb-[8px] pt-[18px] text-left'>
									Description
								</th>
								<th className='pb-[8px] pt-[18px] text-left'>
									UserName
								</th>
								<th className='pb-[8px] pt-[18px] text-left'>
									Email
								</th>
								<th className='pb-[8px] pt-[18px] text-left'>
									<p className='pr-1'>Actions</p>
								</th>
							</tr>
						</thead>
						<tbody>
							{state.requests.map(element => (
								<tr className='my-[10px] cursor-pointer  rounded-xl bg-white hover:bg-[#EBEBEB]'>
									<td className='pl-3'>
										<div className='py-2 '>
											<div className='flex h-[50px] w-[50px] items-center justify-center overflow-hidden rounded-full'>
												<img
													src={
														element.user.picture !==
														null
															? `${HOSTNAME}users/picture?path=${element.user.picture}`
															: UserOne
													}
												/>
											</div>
										</div>
									</td>
									<td className='font-medium'>
										<p>{state.details}</p>
									</td>
									<td className='font-medium'>
										<p className='pr-1'>
											{element.user.username}
										</p>
									</td>
									<td className='hidden font-medium md:table-cell'>
										<p className='pr-1'>
											{element.user.email}
										</p>
									</td>
									<td>
										<div className='justify-center'>
											{element.action ? (
												<>
													{element.accept ? (
														<p className='pr-1 text-[black] font-bold'>
															Accepted
														</p>
													) : (
														<p className='pr-1 text-[black] font-bold'>
															Rejected
														</p>
													)}
												</>
											) : (
												<>
													<button
														className='mx-[10px] h-[22px] w-[50px] justify-center rounded-[10px] bg-[#FF8216]  text-[15px]  text-white hover:bg-[#F7A541] active:bg-[#FFD583] sm:h-[35px] sm:w-[200px]'
														style={{
															transition: '0.3s',
														}}
														onClick={() =>
															acceptRequest(
																state._id,
																element.user._id
															)
														}>
														Accept
													</button>
													<button
														className='mx-[10px] h-[22px] w-[50px] justify-center rounded-[10px] bg-[#FF8216]  text-[15px]  text-white hover:bg-[#F7A541] active:bg-[#FFD583] sm:h-[35px] sm:w-[200px]'
														style={{
															transition: '0.3s',
														}}
														onClick={() =>
															rejectRequest(
																state._id,
																element.user._id
															)
														}>
														Reject
													</button>
												</>
											)}
										</div>
									</td>
								</tr>
							))}
						</tbody>
					</table>
				</div>
				<ToastContainer />
			</div>
		);
	} else {
		return (
			<div className='flex w-full flex-col'>
				{/* Top Area */}
				<div className='flex w-full flex-col justify-between'>
					<h1 className='text-[20px] font-bold'>Job Details</h1>
				</div>

				{/* Description Card */}
				<div className='my-[22px] flex  w-full flex-row flex-wrap  rounded-xl bg-[white] shadow-xl '>
					<div className=' mb-[10px]  flex h-[80px] w-full  flex-row items-center justify-between rounded-xl  bg-[white] p-4 '>
						<div className='avatar items-center'>
							<div className='w-[50px] rounded-full'>
								<img
									src={
										state.creator.picture !== null
											? `${HOSTNAME}users/picture?path=${state.creator.picture}`
											: UserOne
									}
								/>
							</div>
						</div>

						<div className='mr-[auto]  px-5'>
							<p className='text-[14px] font-bold text-[black] sm:text-[24px]'>
								{state.creator.username}
								<span className='ml-[10px] text-[12px] text-[black]'>
									{state.creator.email}
								</span>
							</p>
							<p className='text-[12px] text-[black]'>
								{state.expiry_date}
							</p>
						</div>

						{/* Buttons */}
						<div className=' ml-[auto] flex flex-row items-center justify-between '>
							<button
								className='mx-[10px] h-[22px] w-[100px] justify-center rounded-[10px] bg-[#b0b3b3]   text-[15px]  text-white  active:bg-[#FFD583] sm:h-[35px] sm:w-[200px]'
								style={{ transition: '0.3s' }}
								onClick={stopRequest}
								disabled={true}>
								{receiving
									? 'Stop Receving'
									: 'Start Receiving'}
							</button>
							<button
								className='h-[22px] w-[100px] justify-center rounded-[10px] bg-[#b0b3b3]  text-[15px]   text-white active:bg-[#FFD583] sm:h-[35px] sm:w-[200px]'
								style={{ transition: '0.3s' }}
								disabled={true}>
								Delete Job
							</button>
						</div>
					</div>

					<div className='mb-[20px]  flex w-full flex-row justify-between rounded-xl bg-[white] px-4'>
						<div className='w-[80%]'>
							<p className='text-[14px] text-[black] '>
								{state.details}
							</p>
							<p className='text-[12px] text-[black]'>
								Posted On:
								<span>{format(state.created_at)}</span>
							</p>
						</div>
						<div className='w-[20%] flex items-center justify-center'>
							<p className=' text-[24px] font-bold text-[black]'>
								{state.price}
							</p>
						</div>
					</div>
				</div>

				{/*Heading */}
				<Heading text={'Job Candidates'} />

				{/* Job Candidates Table */}
				<div className='my-[22px] w-[full] overflow-x-auto'>
					<table className='w-full table-auto rounded-xl bg-[white] text-[12px] font-normal text-text  md:text-[14px]'>
						<thead>
							<tr className='border-b-[3px] border-[#E4E4E4]'>
								<th className='pb-[8px] pl-4 pt-[18px] text-left'>
									Picture
								</th>
								<th className='pb-[8px] pt-[18px] text-left'>
									Description
								</th>
								<th className='pb-[8px] pt-[18px] text-left'>
									UserName
								</th>
								<th className='pb-[8px] pt-[18px] text-left'>
									Email
								</th>
								<th className='pb-[8px] pt-[18px] text-left'>
									<p className='pr-1'>Actions</p>
								</th>
							</tr>
						</thead>
						<tbody>
							{state.requests.map(element => (
								<tr className='my-[10px] cursor-pointer  rounded-xl bg-white hover:bg-[#EBEBEB]'>
									<td className='pl-3'>
										<div className='py-2 '>
											<div className='flex h-[50px] w-[50px] items-center justify-center overflow-hidden rounded-full'>
												<img
													src={
														element.user.picture !==
														null
															? `${HOSTNAME}users/picture?path=${element.user.picture}`
															: UserOne
													}
												/>
											</div>
										</div>
									</td>
									<td className='font-medium'>
										<p>{state.details}</p>
									</td>
									<td className='font-medium'>
										<p className='pr-1'>
											{element.user.username}
										</p>
									</td>
									<td className='hidden font-medium md:table-cell'>
										<p className='pr-1'>
											{element.user.email}
										</p>
									</td>
									<td>
										<div className='justify-center'>
											{element.action ? (
												<>
													{element.accept ? (
														<p className='pr-1'>
															Accepted
														</p>
													) : (
														<p className='pr-1'>
															Accepted
														</p>
													)}
												</>
											) : (
												<>
													<p className='pr-1'>
														No Action Taken
													</p>
												</>
											)}
										</div>
									</td>
								</tr>
							))}
						</tbody>
					</table>
				</div>
			</div>
		);
	}
};

export default JobDetails;
