import UserOne from '../../assets/img1.png';
import { useNavigate } from 'react-router-dom';
import { HOSTNAME } from '../../../Config';

const TopCreators = ({ data, num1, num2 }) => {
	const navigate = useNavigate();

	return (
		<table className='w-[49%] table-auto  text-[12px] font-normal text-text  md:text-[14px] '>
			<tbody>
				{data.slice(num1, num2).map((element, index) => (
					<tr key={index} className='h-[90px] bg-[transparent]'>
						<td
							onClick={() =>
								navigate(`/otherProfile`, {
									state: element,
								})
							}>
							<div className=' mb-[20px]  flex h-[80px] cursor-pointer flex-row items-center justify-between  rounded-[20px] bg-[white]  p-4  shadow-xl hover:bg-[#EBEBEB]'>
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
	);
};

export default TopCreators;
