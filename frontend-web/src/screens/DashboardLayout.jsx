import { useState, useEffect } from 'react';
import Sidebar from '../Components/Sidebar';
import { Outlet } from 'react-router-dom';
import Header from '../Components/Header';
import { getUser } from '../api/Api';
import { useDispatch } from 'react-redux';
import { setUser } from '../redux/userSlice';

const DashboardLayout = () => {
	const [sidebarOpen, setSidebarOpen] = useState(false);
	const dispatch = useDispatch();
	const fetchUsers = async () => {
		const data = await getUser();
		dispatch(setUser(data.user));
	};

	useEffect(() => {
		fetchUsers();
	}, []);

	return (
		<div className='relative'>
			{/* <!-- ===== Page Wrapper Start ===== --> */}
			<div className='flex h-screen'>
				{/* <!-- ===== Sidebar Start ===== --> */}
				<Sidebar
					sidebarOpen={sidebarOpen}
					setSidebarOpen={setSidebarOpen}
				/>
				{/* <!-- ===== Sidebar End ===== --> */}

				{/* <!-- ===== Content Area Start ===== --> */}
				<div className='relative flex flex-1 flex-col overflow-y-auto overflow-x-hidden bg-[#F2F2F2]'>
					{/* <!-- ===== Header Start ===== --> */}
					<Header
						sidebarOpen={sidebarOpen}
						setSidebarOpen={setSidebarOpen}
					/>
					{/* <!-- ===== Header End ===== --> */}

					{/* <!-- ===== Main Content Start ===== --> */}
					<main>
						<div className='mx-auto max-w-screen-2xl p-4 md:p-6 2xl:p-10'>
							<Outlet />
						</div>
					</main>
					{/* <!-- ===== Main Content End ===== --> */}
				</div>
				{/* <!-- ===== Content Area End ===== --> */}
			</div>
			{/* <!-- ===== Page Wrapper End ===== --> */}
		</div>
	);
};

export default DashboardLayout;
