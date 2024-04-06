import UserOne from '../../assets/img1.png';
import { useNavigate } from 'react-router-dom';
import { HOSTNAME } from '../../../Config';

const SideTable = ({ data }) => {
	const navigate = useNavigate();
	return (
		<div className='my-[22px] flex w-[full]  justify-between overflow-x-auto'>
			<table className='w-full table-auto  text-[12px] font-normal text-text  md:text-[14px] '>
				<tbody>
					{data.slice(0, 3).map((element, index) => (
						<tr key={index}>
							<td>
								<div
									className=' mb-[20px]  flex h-[80px] cursor-pointer flex-row items-center justify-between  rounded-[20px] bg-[white]  p-4  shadow-xl hover:bg-[#EBEBEB]'
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
							</td>
						</tr>
					))}
				</tbody>
			</table>
		</div>
	);
};

export default SideTable;
